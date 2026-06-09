"""
Utilities for lightweight performance/environment checks.
"""

import os
import warnings
from contextlib import contextmanager
from typing import Iterable, Optional, Set

_blas_warning_emitted = False


@contextmanager
def numba_thread_limit(cpu: Optional[int]):
    """Cap numba's parallel kernels to ``cpu`` threads for the duration.

    numba's ``prange`` kernels otherwise use every core detected at import,
    ignoring PANICLE's ``--ncpus`` budget. This pins the runtime thread count to
    the caller's resolved CPU count (already 0 -> all-cores expanded upstream)
    and restores the previous setting on exit. No-op when numba is absent or
    ``cpu`` is None/<=0 (leave numba's default).
    """
    if cpu is None or int(cpu) <= 0:
        yield
        return
    try:
        import numba
    except ImportError:
        yield
        return
    try:
        max_threads = int(numba.config.NUMBA_NUM_THREADS)
        target = max(1, min(int(cpu), max_threads))
        prev = numba.get_num_threads()
        numba.set_num_threads(target)
    except Exception:  # pragma: no cover - defensive; fall back to default
        yield
        return
    try:
        yield
    finally:
        try:
            numba.set_num_threads(prev)
        except Exception:  # pragma: no cover
            pass


def available_cpu_count() -> int:
    """Number of CPUs this process may actually use.

    Prefers the scheduler affinity mask (``os.sched_getaffinity``), which
    respects cgroup/cpuset limits and HPC scheduler allocations (SLURM
    ``--cpus-per-task``, etc.). Falls back to ``os.cpu_count()`` on platforms
    without affinity support (macOS/Windows). Always returns at least 1.

    Use this instead of ``os.cpu_count()`` / ``multiprocessing.cpu_count()``
    when expanding a ``cpu=0`` ("all cores") request, so PANICLE never spins up
    more threads than its allocation on a shared node.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux platforms
        return max(1, os.cpu_count() or 1)


def _normalise_tokens(values: Iterable[object]) -> Set[str]:
    """Flatten nested iterables of strings into a lowercase token set."""

    tokens: Set[str] = set()

    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            tokens.add(value.lower())
            continue
        if isinstance(value, (bytes, bytearray)):
            tokens.add(value.decode("utf-8", errors="ignore").lower())
            continue
        if isinstance(value, dict):
            tokens.update(_normalise_tokens(value.values()))
            continue
        if isinstance(value, Iterable):
            tokens.update(_normalise_tokens(value))
    return tokens


def _detect_blas_backend() -> Optional[str]:
    """Return a lowercase name for the detected BLAS backend, if possible."""

    try:
        from numpy import __config__ as np_config  # type: ignore
    except Exception:
        return None

    info_keys = (
        "blas_opt_info",
        "openblas_info",
        "openblas_ilp64_info",
        "blas_mkl_info",
        "mkl_info",
        "accelerate_info",
    )

    for key in info_keys:
        try:
            info = np_config.get_info(key)
        except Exception:
            info = {}
        if not info:
            continue
        tokens = _normalise_tokens(info.values())
        if not tokens:
            tokens = {str(info).lower()}
        joined = " ".join(sorted(tokens))
        if "openblas" in joined:
            return "openblas"
        if "mkl" in joined or "intel" in joined:
            return "mkl"
        if "accelerate" in joined or "-framework accelerate" in joined:
            return "accelerate"
        if "blis" in joined:
            return "blis"
        if "atlas" in joined:
            return "atlas"
    return None


def warn_if_potential_single_thread_blas() -> None:
    """
    Emit a guidance warning when NumPy appears to be linked against a BLAS that
    defaults to single-thread execution (e.g. Apple's Accelerate without thread
    overrides). The warning fires at most once per process.
    """

    global _blas_warning_emitted

    if _blas_warning_emitted or os.environ.get("PYMVP_SKIP_BLAS_CHECK"):
        return

    backend = _detect_blas_backend()
    if backend is None:
        return

    if backend == "accelerate":
        # Accelerate typically runs single-threaded unless VECLIB_MAXIMUM_THREADS
        # is explicitly set. Guide the user towards OpenBLAS or configuring
        # veclib.
        if not os.environ.get("VECLIB_MAXIMUM_THREADS"):
            warnings.warn(
                (
                    "PANICLE detected that NumPy is linked against Apple's Accelerate "
                    "BLAS/LAPACK. This backend defaults to single-core execution, "
                    "which can severely slow FarmCPU and MLM scans. For multi-core "
                    "performance either reinstall NumPy/SciPy with an OpenBLAS backend "
                    "(e.g. `brew install openblas` followed by "
                    "`pip install --no-binary=:all: numpy scipy`) or set "
                    "`VECLIB_MAXIMUM_THREADS=<num_cores>` before running PANICLE."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _blas_warning_emitted = True
            return

    if backend == "blis":
        # BLIS defaults to single-thread on macOS wheels; provide similar guidance.
        if not os.environ.get("OMP_NUM_THREADS"):
            warnings.warn(
                (
                    "PANICLE detected a BLIS backend for NumPy/SciPy without an explicit "
                    "OMP_NUM_THREADS setting. If analyses only use a single core, "
                    "consider installing OpenBLAS or setting OMP_NUM_THREADS to the "
                    "desired number of threads."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _blas_warning_emitted = True
            return

    if backend in {"atlas"}:
        warnings.warn(
            (
                "PANICLE detected that NumPy is using the ATLAS BLAS backend. ATLAS "
                "is typically single-threaded on modern macOS builds; if performance "
                "is limited to one core, consider moving to OpenBLAS or MKL."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        _blas_warning_emitted = True
        return

    # Assume OpenBLAS/MKL/etc. are multi-thread ready; no warning needed.
