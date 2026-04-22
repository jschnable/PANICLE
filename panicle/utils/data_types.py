"""
Core data structures for PANICLE package
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any, List, Sequence
from pathlib import Path


MARKER_ID_COLUMN = "MARKER"
LEGACY_MARKER_ID_COLUMN = "SNP"
CHROM_COLUMN = "CHROM"
POS_COLUMN = "POS"

_MARKER_COLUMN_ALIASES = ("MARKER", "marker", "Marker", "SNP", "snp", "rs", "RS")
_CHROM_COLUMN_ALIASES = ("CHROM", "Chr", "chr", "chromosome", "CHR")
_POS_COLUMN_ALIASES = ("POS", "Pos", "pos", "position", "bp")


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def infer_marker_id_column(columns: Sequence[str]) -> Optional[str]:
    """Return the first marker-ID column name found in ``columns``."""
    return _first_present(columns, _MARKER_COLUMN_ALIASES)


def canonicalize_genotype_map_dataframe(
    df: pd.DataFrame,
    *,
    include_legacy_snp_alias: bool = True,
) -> pd.DataFrame:
    """Normalize map column names to PANICLE's canonical marker schema.

    Canonical required columns are ``MARKER``, ``CHROM``, and ``POS``.
    For compatibility, the legacy ``SNP`` alias can be retained.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Genotype map data must be a pandas DataFrame")

    out = df.copy()
    cols = list(out.columns)

    marker_col = _first_present(cols, _MARKER_COLUMN_ALIASES)
    chrom_col = _first_present(cols, _CHROM_COLUMN_ALIASES)
    pos_col = _first_present(cols, _POS_COLUMN_ALIASES)

    if marker_col is None:
        raise ValueError(
            f"Missing required marker ID column. Accepted aliases: {list(_MARKER_COLUMN_ALIASES)}"
        )
    if chrom_col is None:
        raise ValueError(
            f"Missing required chromosome column. Accepted aliases: {list(_CHROM_COLUMN_ALIASES)}"
        )
    if pos_col is None:
        raise ValueError(
            f"Missing required position column. Accepted aliases: {list(_POS_COLUMN_ALIASES)}"
        )

    rename_map: Dict[str, str] = {}
    if marker_col != MARKER_ID_COLUMN:
        rename_map[marker_col] = MARKER_ID_COLUMN
    if chrom_col != CHROM_COLUMN:
        rename_map[chrom_col] = CHROM_COLUMN
    if pos_col != POS_COLUMN:
        rename_map[pos_col] = POS_COLUMN
    if rename_map:
        out = out.rename(columns=rename_map)

    if LEGACY_MARKER_ID_COLUMN in out.columns:
        marker_equal = out[MARKER_ID_COLUMN].astype(str).equals(
            out[LEGACY_MARKER_ID_COLUMN].astype(str)
        )
        if not marker_equal:
            raise ValueError(
                f"Columns '{MARKER_ID_COLUMN}' and '{LEGACY_MARKER_ID_COLUMN}' contain different values"
            )
    elif include_legacy_snp_alias:
        out[LEGACY_MARKER_ID_COLUMN] = out[MARKER_ID_COLUMN].astype(str)

    base_cols = [MARKER_ID_COLUMN, CHROM_COLUMN, POS_COLUMN]
    if LEGACY_MARKER_ID_COLUMN in out.columns:
        base_cols.append(LEGACY_MARKER_ID_COLUMN)
    remaining_cols = [c for c in out.columns if c not in base_cols]
    return out[base_cols + remaining_cols]


def group_marker_indices_by_labels(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ordered marker indices grouped by chromosome/label."""
    values = np.asarray(labels).astype(str, copy=False)
    if values.ndim != 1:
        raise ValueError("Marker labels must be a 1D array")
    if values.size == 0:
        return {}

    unique_values, inverse_indices = np.unique(values, return_inverse=True)
    sorted_order = np.argsort(inverse_indices, kind="stable")
    sorted_inverse = inverse_indices[sorted_order]
    boundaries = np.concatenate(
        [[0], np.where(np.diff(sorted_inverse) != 0)[0] + 1, [values.size]]
    )

    grouped: Dict[str, np.ndarray] = {}
    for idx, label in enumerate(unique_values):
        start, end = boundaries[idx], boundaries[idx + 1]
        grouped[str(label)] = sorted_order[start:end]
    return grouped


def attach_genotype_map_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach reusable chromosome-group metadata to a canonical map DataFrame."""
    attrs = getattr(df, "attrs", {})
    if attrs.get("chromosome_groups") and attrs.get("chromosome_order"):
        return df

    chrom_values = np.asarray(df[CHROM_COLUMN]).astype(str, copy=False)
    chrom_groups = group_marker_indices_by_labels(chrom_values)
    df.attrs["chromosome_groups"] = chrom_groups
    df.attrs["chromosome_order"] = list(chrom_groups.keys())
    return df


def _pack_chromosome_groups(
    chrom_groups: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.array(list(chrom_groups.keys()), dtype=str)
    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    for idx, chrom in enumerate(order):
        offsets[idx + 1] = offsets[idx] + int(len(chrom_groups[str(chrom)]))
    flat_indices = np.empty(int(offsets[-1]), dtype=np.int64)
    for idx, chrom in enumerate(order):
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        flat_indices[start:end] = np.asarray(chrom_groups[str(chrom)], dtype=np.int64)
    return order, offsets, flat_indices


def _unpack_chromosome_groups(
    order: Sequence[str],
    offsets: np.ndarray,
    flat_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    chrom_groups: Dict[str, np.ndarray] = {}
    for idx, chrom in enumerate(order):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        chrom_groups[str(chrom)] = np.asarray(flat_indices[start:end], dtype=np.int64)
    return chrom_groups


class _PackedUtf8Column:
    """Lazy UTF-8 string column stored as a byte blob plus offsets."""

    def __init__(self, offsets: np.ndarray, data: np.ndarray):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.uint8)
        self._decoded: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return max(0, int(self.offsets.size) - 1)

    def to_numpy(self) -> np.ndarray:
        if self._decoded is None:
            buffer = memoryview(self.data)
            out = np.empty(len(self), dtype=object)
            for idx in range(len(out)):
                start = int(self.offsets[idx])
                end = int(self.offsets[idx + 1])
                out[idx] = bytes(buffer[start:end]).decode("utf-8")
            self._decoded = out
        return self._decoded


class _CategoricalUtf8Column:
    """Lazy low-cardinality string column stored as category codes."""

    def __init__(self, codes: np.ndarray, categories: np.ndarray):
        self.codes = np.asarray(codes, dtype=np.int32)
        self.categories = np.asarray(categories, dtype=object)
        self._decoded: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.codes.size)

    def to_numpy(self) -> np.ndarray:
        if self._decoded is None:
            self._decoded = np.asarray(self.categories[self.codes], dtype=object)
        return self._decoded


def _pack_utf8_column(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a string column as UTF-8 bytes with offsets."""
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    payload = bytearray()
    for idx, value in enumerate(values):
        encoded = str(value).encode("utf-8")
        payload.extend(encoded)
        offsets[idx + 1] = len(payload)
    data = np.frombuffer(bytes(payload), dtype=np.uint8)
    return offsets, data


def _materialize_lazy_column(
    column: Union[np.ndarray, "_PackedUtf8Column", "_CategoricalUtf8Column"]
) -> np.ndarray:
    if hasattr(column, "to_numpy"):
        return column.to_numpy()  # type: ignore[return-value]
    return np.asarray(column)


def save_genotype_map_cache(
    cache_path: Union[str, Path],
    map_data: Union[pd.DataFrame, "GenotypeMap"],
) -> None:
    """Persist genotype-map metadata to a fast binary cache."""
    if isinstance(map_data, GenotypeMap):
        map_df = map_data.to_dataframe()
        metadata = map_data.metadata
    else:
        map_df = map_data.copy()
        metadata = getattr(map_df, "attrs", {})

    map_df = canonicalize_genotype_map_dataframe(map_df)
    map_df = attach_genotype_map_metadata(map_df)

    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(2, dtype=np.int16),
        "columns": np.asarray(map_df.columns, dtype=str),
        "is_imputed": np.asarray(bool(metadata.get("is_imputed", map_df.attrs.get("is_imputed", False)))),
    }
    column_kinds = np.empty(len(map_df.columns), dtype="<U24")
    alias_of = np.full(len(map_df.columns), -1, dtype=np.int64)

    for idx, column in enumerate(map_df.columns):
        series = map_df[column]
        values = series.to_numpy(copy=False)
        if (
            column == LEGACY_MARKER_ID_COLUMN
            and MARKER_ID_COLUMN in map_df.columns
            and series.astype(str).equals(map_df[MARKER_ID_COLUMN].astype(str))
        ):
            column_kinds[idx] = "alias"
            alias_of[idx] = int(list(map_df.columns).index(MARKER_ID_COLUMN))
            continue

        if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_numeric_dtype(series.dtype):
            column_kinds[idx] = "numeric"
            arrays[f"col_{idx}"] = np.asarray(values)
            continue

        string_values = series.astype(str).to_numpy(dtype=object, copy=False)
        if column == CHROM_COLUMN:
            categories, codes = np.unique(string_values, return_inverse=True)
            column_kinds[idx] = "categorical_utf8"
            arrays[f"col_{idx}_codes"] = np.asarray(codes, dtype=np.int32)
            arrays[f"col_{idx}_categories"] = np.asarray(categories, dtype=str)
            continue

        column_kinds[idx] = "packed_utf8"
        offsets, data = _pack_utf8_column(string_values)
        arrays[f"col_{idx}_offsets"] = offsets
        arrays[f"col_{idx}_data"] = data

    arrays["column_kinds"] = column_kinds
    arrays["column_alias_of"] = alias_of

    chrom_groups = map_df.attrs.get("chromosome_groups") or metadata.get("chromosome_groups")
    if chrom_groups is None:
        chrom_groups = group_marker_indices_by_labels(
            np.asarray(map_df[CHROM_COLUMN]).astype(str, copy=False)
        )
    order, offsets, flat_indices = _pack_chromosome_groups(chrom_groups)
    arrays["chrom_group_keys"] = order
    arrays["chrom_group_offsets"] = offsets
    arrays["chrom_group_indices"] = flat_indices

    np.savez(cache_path, **arrays)


def load_genotype_map_cache(
    cache_path: Union[str, Path],
    *,
    legacy_csv_path: Optional[Union[str, Path]] = None,
    migrate_legacy: bool = False,
    legacy_is_imputed: Optional[bool] = None,
) -> "GenotypeMap":
    """Load genotype-map metadata from a binary cache or legacy CSV cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if "format_version" in archive and "column_kinds" in archive:
                columns = archive["columns"].astype(str).tolist()
                column_kinds = archive["column_kinds"].astype(str).tolist()
                if "column_alias_of" in archive:
                    alias_of = np.asarray(archive["column_alias_of"], dtype=np.int64)
                else:
                    alias_of = np.full(len(columns), -1, dtype=np.int64)
                column_data: Dict[str, Any] = {}
                for idx, column in enumerate(columns):
                    kind = column_kinds[idx]
                    if kind == "alias":
                        source_idx = int(alias_of[idx])
                        if source_idx < 0:
                            raise ValueError(f"Invalid alias mapping for cached column '{column}'")
                        column_data[column] = column_data[columns[source_idx]]
                    elif kind == "numeric":
                        column_data[column] = archive[f"col_{idx}"]
                    elif kind == "packed_utf8":
                        column_data[column] = _PackedUtf8Column(
                            archive[f"col_{idx}_offsets"],
                            archive[f"col_{idx}_data"],
                        )
                    elif kind == "categorical_utf8":
                        column_data[column] = _CategoricalUtf8Column(
                            archive[f"col_{idx}_codes"],
                            archive[f"col_{idx}_categories"],
                        )
                    else:
                        raise ValueError(f"Unknown cached column kind: {kind}")

                metadata: Dict[str, Any] = {}
                if "is_imputed" in archive:
                    metadata["is_imputed"] = bool(np.asarray(archive["is_imputed"]).item())
                if (
                    "chrom_group_keys" in archive
                    and "chrom_group_offsets" in archive
                    and "chrom_group_indices" in archive
                ):
                    order = archive["chrom_group_keys"].astype(str).tolist()
                    offsets = np.asarray(archive["chrom_group_offsets"], dtype=np.int64)
                    flat = np.asarray(archive["chrom_group_indices"], dtype=np.int64)
                    metadata["chromosome_order"] = order
                    metadata["chromosome_groups"] = _unpack_chromosome_groups(
                        order,
                        offsets,
                        flat,
                    )
                return GenotypeMap.from_columns(
                    column_data,
                    column_order=columns,
                    metadata=metadata,
                )

            if legacy_csv_path is None:
                columns = archive["columns"].astype(str).tolist()
                frame_data = {
                    column: archive[f"col_{idx}"]
                    for idx, column in enumerate(columns)
                }
                map_df = canonicalize_genotype_map_dataframe(pd.DataFrame(frame_data, copy=False))
                attach_genotype_map_metadata(map_df)
                if "is_imputed" in archive:
                    map_df.attrs["is_imputed"] = bool(np.asarray(archive["is_imputed"]).item())
                return GenotypeMap(map_df, metadata=dict(map_df.attrs))

    if legacy_csv_path is None:
        raise FileNotFoundError(f"Genotype map cache not found: {cache_path}")

    legacy_path = Path(legacy_csv_path)
    if not legacy_path.exists():
        raise FileNotFoundError(f"Genotype map cache not found: {cache_path}")

    map_df = canonicalize_genotype_map_dataframe(pd.read_csv(legacy_path))
    map_df = attach_genotype_map_metadata(map_df)
    if legacy_is_imputed is not None:
        map_df.attrs["is_imputed"] = bool(legacy_is_imputed)
    if migrate_legacy:
        try:
            save_genotype_map_cache(cache_path, map_df)
        except Exception:
            pass
    return GenotypeMap(map_df, metadata=dict(map_df.attrs))


def _read_table_with_auto_separator(path: Union[str, Path], *, header: Optional[int]) -> pd.DataFrame:
    """Read a small tabular file while auto-detecting common separators."""
    return pd.read_csv(path, sep=None, engine="python", header=header)


def _normalize_phenotype_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize phenotype columns while preserving user-supplied trait names."""
    if df.shape[1] < 2:
        raise ValueError(
            f"Phenotype must have at least 2 columns (ID + trait), got {df.shape[1]}"
        )

    out = df.copy()
    default_columns = isinstance(out.columns, pd.RangeIndex) and list(out.columns) == list(range(out.shape[1]))

    if "ID" in out.columns:
        trait_columns = [col for col in out.columns if col != "ID"]
        out = out[["ID"] + trait_columns]
    else:
        first_column = out.columns[0]
        out = out.rename(columns={first_column: "ID"})
        trait_columns = list(out.columns[1:])

    if not trait_columns:
        raise ValueError("Phenotype must include at least one trait column")

    if default_columns:
        if len(trait_columns) == 1:
            trait_names = ["Trait"]
        else:
            trait_names = [f"Trait{i + 1}" for i in range(len(trait_columns))]
        out.columns = ["ID"] + trait_names
        return out

    trait_names = []
    seen_names = {"ID"}
    single_trait = len(trait_columns) == 1
    for idx, column in enumerate(trait_columns, start=1):
        if isinstance(column, str):
            normalized = column.strip()
        elif column is None:
            normalized = ""
        else:
            normalized = str(column)

        if not normalized or normalized == "ID":
            normalized = "Trait" if single_trait else f"Trait{idx}"

        candidate = normalized
        suffix = 2
        while candidate in seen_names:
            candidate = f"{normalized}_{suffix}"
            suffix += 1
        seen_names.add(candidate)
        trait_names.append(candidate)

    out.columns = ["ID"] + trait_names
    return out

class Phenotype:
    """Phenotype data structure compatible with R rMVP format

    Supports single or multiple traits:
    - Column 1: Individual IDs
    - Columns 2+: Trait values (one or more traits)

    Examples:
        Single trait: [ID, Trait1]
        Multiple traits: [ID, Trait1, Trait2, Trait3, Trait4]
    """

    def __init__(self, data: Union[np.ndarray, pd.DataFrame, str, Path]):
        if isinstance(data, (str, Path)):
            try:
                self.data = _read_table_with_auto_separator(data, header=0)
                if self.data.shape[1] < 2:
                    self.data = _read_table_with_auto_separator(data, header=None)
            except Exception:
                self.data = _read_table_with_auto_separator(data, header=None)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be array, DataFrame, or file path")

        self.data = _normalize_phenotype_dataframe(self.data)

    @property
    def ids(self) -> pd.Series:
        """Individual IDs"""
        return self.data['ID']

    @property
    def trait_names(self) -> List[str]:
        """Names of all trait columns"""
        return list(self.data.columns[1:])

    @property
    def values(self) -> pd.DataFrame:
        """All trait values as DataFrame"""
        return self.data.iloc[:, 1:]

    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return len(self.data)

    @property
    def n_traits(self) -> int:
        """Number of traits"""
        return self.data.shape[1] - 1

    def get_trait(self, trait: Union[int, str]) -> pd.Series:
        """Get values for a specific trait by index or name.

        Args:
            trait: Trait index (0-based) or trait column name

        Returns:
            Series of trait values
        """
        if isinstance(trait, int):
            if trait < 0 or trait >= self.n_traits:
                raise IndexError(f"Trait index {trait} out of range (0-{self.n_traits-1})")
            return self.data.iloc[:, trait + 1]
        else:
            if trait not in self.data.columns:
                raise KeyError(f"Trait '{trait}' not found. Available: {self.trait_names}")
            return self.data[trait]

    def get_single_trait_array(self, trait: Union[int, str] = 0) -> np.ndarray:
        """Get ID + single trait as numpy array for GWAS methods.

        Args:
            trait: Trait index (0-based) or trait column name

        Returns:
            Array of shape (n_individuals, 2) with [ID, trait_value]
        """
        trait_values = self.get_trait(trait)
        return np.column_stack([self.ids.values, trait_values.values])

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (all columns including ID)"""
        return self.data.values


class GenotypeMap:
    """Marker map information compatible with R rMVP format.

    Required columns: [MARKER, CHROM, POS]
    Optional columns: [REF, ALT]
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, Path], metadata: Optional[Dict[str, Any]] = None):
        if isinstance(data, (str, Path)):
            raw_df = _read_table_with_auto_separator(data, header=0)
        elif isinstance(data, pd.DataFrame):
            raw_df = data.copy()
        else:
            raise ValueError("Data must be DataFrame or file path")

        attrs = getattr(raw_df, "attrs", {})
        inherited_metadata = {
            key: attrs[key]
            for key in ("chromosome_groups", "chromosome_order", "is_imputed")
            if key in attrs
        }
        if metadata:
            inherited_metadata.update(metadata)
        self.metadata: Dict[str, Any] = inherited_metadata
        self._dataframe_cache: Optional[pd.DataFrame] = canonicalize_genotype_map_dataframe(
            raw_df,
            include_legacy_snp_alias=True,
        )
        attach_genotype_map_metadata(self._dataframe_cache)
        if "is_imputed" in self.metadata:
            self._dataframe_cache.attrs["is_imputed"] = self.metadata["is_imputed"]
        self._column_order = list(self._dataframe_cache.columns)
        self._column_data: Dict[str, Any] = {
            column: self._dataframe_cache[column].to_numpy(copy=False)
            for column in self._column_order
        }

    @classmethod
    def from_columns(
        cls,
        column_data: Dict[str, Any],
        *,
        column_order: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GenotypeMap":
        obj = cls.__new__(cls)
        obj.metadata = dict(metadata) if metadata is not None else {}
        obj._dataframe_cache = None
        obj._column_order = list(column_order) if column_order is not None else list(column_data.keys())
        obj._column_data = dict(column_data)
        if MARKER_ID_COLUMN not in obj._column_data:
            raise ValueError(f"Missing required marker ID column '{MARKER_ID_COLUMN}'")
        if CHROM_COLUMN not in obj._column_data:
            raise ValueError(f"Missing required chromosome column '{CHROM_COLUMN}'")
        if POS_COLUMN not in obj._column_data:
            raise ValueError(f"Missing required position column '{POS_COLUMN}'")
        if LEGACY_MARKER_ID_COLUMN not in obj._column_data:
            obj._column_data[LEGACY_MARKER_ID_COLUMN] = obj._column_data[MARKER_ID_COLUMN]
            if LEGACY_MARKER_ID_COLUMN not in obj._column_order:
                insert_at = obj._column_order.index(POS_COLUMN) + 1 if POS_COLUMN in obj._column_order else 1
                obj._column_order.insert(insert_at, LEGACY_MARKER_ID_COLUMN)
        return obj

    @property
    def data(self) -> pd.DataFrame:
        """Map data as a pandas DataFrame, materialized lazily if needed."""
        if self._dataframe_cache is None:
            frame_data = {
                column: _materialize_lazy_column(self._column_data[column])
                for column in self._column_order
            }
            self._dataframe_cache = pd.DataFrame(frame_data, copy=False)
            if "chromosome_groups" in self.metadata:
                self._dataframe_cache.attrs["chromosome_groups"] = self.metadata["chromosome_groups"]
            if "chromosome_order" in self.metadata:
                self._dataframe_cache.attrs["chromosome_order"] = self.metadata["chromosome_order"]
            if "is_imputed" in self.metadata:
                self._dataframe_cache.attrs["is_imputed"] = self.metadata["is_imputed"]
            attach_genotype_map_metadata(self._dataframe_cache)
        return self._dataframe_cache

    @property
    def attrs(self) -> Dict[str, Any]:
        """Pandas-style attrs exposed for compatibility with DataFrame callers."""
        attrs = self.data.attrs
        if "is_imputed" in self.metadata:
            attrs["is_imputed"] = self.metadata["is_imputed"]
        chrom_groups = self.metadata.get("chromosome_groups")
        if chrom_groups is not None:
            attrs["chromosome_groups"] = chrom_groups
        elif "chromosome_groups" not in attrs:
            attrs["chromosome_groups"] = group_marker_indices_by_labels(
                np.asarray(self.chromosomes).astype(str, copy=False)
            )
        chrom_order = self.metadata.get("chromosome_order")
        if chrom_order is not None:
            attrs["chromosome_order"] = chrom_order
        elif "chromosome_order" not in attrs:
            attrs["chromosome_order"] = list(attrs["chromosome_groups"].keys())
        return attrs

    def _get_column(self, column: str) -> np.ndarray:
        return _materialize_lazy_column(self._column_data[column])
    
    @property
    def marker_ids(self) -> pd.Series:
        """Marker identifiers."""
        return pd.Series(self._get_column(MARKER_ID_COLUMN), name=MARKER_ID_COLUMN)

    @property
    def snp_ids(self) -> pd.Series:
        """Legacy alias for marker identifiers."""
        return self.marker_ids
    
    @property
    def chromosomes(self) -> pd.Series:
        """Chromosome numbers"""
        return pd.Series(self._get_column(CHROM_COLUMN), name=CHROM_COLUMN)
    
    @property
    def positions(self) -> pd.Series:
        """Physical positions"""
        return pd.Series(self._get_column(POS_COLUMN), name=POS_COLUMN)
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self._column_data[POS_COLUMN])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        out = self.data.copy()
        if "chromosome_groups" in self.metadata:
            out.attrs["chromosome_groups"] = self.metadata["chromosome_groups"]
        if "chromosome_order" in self.metadata:
            out.attrs["chromosome_order"] = self.metadata["chromosome_order"]
        if "is_imputed" in self.metadata:
            out.attrs["is_imputed"] = self.metadata["is_imputed"]
        return out

    def with_metadata(self, **metadata: Any) -> "GenotypeMap":
        """Return a new GenotypeMap with merged metadata dictionary."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return GenotypeMap.from_columns(
            self._column_data,
            column_order=self._column_order,
            metadata=merged,
        )

    def subset_markers(self, indices: Union[np.ndarray, list]) -> "GenotypeMap":
        """Return a GenotypeMap restricted to the given marker indices.

        Accepts a boolean mask (length n_markers) or an integer index array.
        Chromosome groups/order are dropped from metadata because they are
        indexed against the original marker layout; callers that need them
        for the subset should recompute after.
        """
        n = self.n_markers
        if isinstance(indices, list):
            indices = np.asarray(indices)
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            if indices.ndim != 1 or indices.size != n:
                raise ValueError("Boolean marker indexer must match the number of markers")
            idx = np.flatnonzero(indices).astype(np.int64, copy=False)
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("Marker indices must be a 1D array-like")
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise IndexError("Marker indices are out of bounds")

        new_columns: Dict[str, Any] = {}
        for col_name in self._column_order:
            data = _materialize_lazy_column(self._column_data[col_name])
            arr = np.asarray(data)
            new_columns[col_name] = arr[idx] if idx.size else arr[:0]

        metadata = {k: v for k, v in self.metadata.items()
                    if k not in ("chromosome_groups", "chromosome_order")}
        return GenotypeMap.from_columns(
            new_columns,
            column_order=self._column_order,
            metadata=metadata,
        )

    def get_chromosome_groups(self) -> Dict[str, np.ndarray]:
        """Return cached chromosome-to-marker index groups."""
        chrom_groups = self.metadata.get("chromosome_groups")
        if chrom_groups is None:
            chrom_groups = self.data.attrs.get("chromosome_groups")
        if chrom_groups is None:
            chrom_groups = group_marker_indices_by_labels(
                np.asarray(self.chromosomes).astype(str, copy=False)
            )
        return chrom_groups

    def get_chromosome_order(self) -> List[str]:
        """Return chromosome labels in cached group order."""
        order = self.metadata.get("chromosome_order")
        if order is None:
            order = self.data.attrs.get("chromosome_order")
        if order is None:
            order = list(self.get_chromosome_groups().keys())
        return list(order)


def impute_major_allele_inplace(geno: np.ndarray, missing_value: int = -9) -> int:
    """Fill missing values in-place with the per-marker major allele (0/1/2)."""
    if geno.size == 0:
        return 0
    missing_mask = geno == missing_value
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return 0
    counts_0 = np.sum(geno == 0, axis=0)
    counts_1 = np.sum(geno == 1, axis=0)
    counts_2 = np.sum(geno == 2, axis=0)
    major_alleles = np.argmax(np.stack([counts_0, counts_1, counts_2], axis=0), axis=0)
    major_alleles = major_alleles.astype(geno.dtype, copy=False)
    geno[missing_mask] = np.broadcast_to(major_alleles, geno.shape)[missing_mask]
    return n_missing


def impute_numpy_batch_major_allele(
    batch: np.ndarray,
    *,
    fill_value: Optional[float] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Impute -9/NaN values for a numpy genotype batch.

    This mirrors the major-allele strategy used by ``GenotypeMatrix`` imputation
    paths so callers get consistent behavior regardless of genotype container type.

    Args:
        batch: Raw genotype slice with shape (n_individuals, n_markers_in_batch).
        fill_value: Optional constant used to replace missing values. If None,
            per-marker major allele is used.
        dtype: Output dtype for the returned array.
    """
    out_dtype = np.dtype(dtype)
    G = np.array(batch, dtype=out_dtype, copy=True)

    if G.size == 0:
        return G

    missing = (G == -9) | np.isnan(G)
    if not missing.any():
        return G

    if fill_value is not None:
        G[missing] = out_dtype.type(fill_value)
        return G

    # Fast path for canonical diploid encoding {0,1,2,-9,NaN}.
    valid_set_mask = (G == 0) | (G == 1) | (G == 2) | missing
    if np.all(valid_set_mask):
        with np.errstate(invalid="ignore"):
            c0 = np.sum(G == 0, axis=0, dtype=np.int32)
            c1 = np.sum(G == 1, axis=0, dtype=np.int32)
            c2 = np.sum(G == 2, axis=0, dtype=np.int32)
        counts = np.stack([c0, c1, c2], axis=0)
        major = np.argmax(counts, axis=0).astype(out_dtype, copy=False)
        G[missing] = np.broadcast_to(major, G.shape)[missing]
        return G

    # Fallback for unexpected genotype coding: pick the mode among non-missing.
    n_markers = G.shape[1]
    for j in range(n_markers):
        col = G[:, j]
        miss = missing[:, j]
        if not miss.any():
            continue
        non_missing = col[~miss]
        if non_missing.size == 0:
            maj = out_dtype.type(0.0)
        else:
            vals, cnts = np.unique(non_missing, return_counts=True)
            maj = out_dtype.type(vals[int(np.argmax(cnts))])
        col[miss] = maj
        G[:, j] = col

    return G


class GenotypeMatrix:
    """Memory-efficient genotype matrix with lazy loading support

    Handles large genotype matrices that may not fit in memory.
    Compatible with R rMVP memory-mapped format.
    Includes pre-computed major alleles for efficient missing data imputation.

    Supports column-major (transposed) storage for efficient individual subsetting.
    When transposed=True, internal data is stored as (n_markers, n_individuals)
    but the API still presents it as (n_individuals, n_markers).
    """

    def __init__(self, data: Union[np.ndarray, str, Path],
                 shape: Optional[Tuple[int, int]] = None,
                 dtype: np.dtype = np.int8,
                 precompute_alleles: bool = True,
                 is_imputed: bool = False,
                 transposed: bool = False,
                 row_indexer: Optional[np.ndarray] = None):
        """Initialize GenotypeMatrix.

        Args:
            data: Numpy array, memmap, or path to memory-mapped file
            shape: Shape for memory-mapped files (n_individuals, n_markers)
                   or (n_markers, n_individuals) if transposed=True
            dtype: Data type for memory-mapped files
            precompute_alleles: Pre-compute major alleles for imputation
            is_imputed: Whether data has been pre-imputed (no -9 values)
            transposed: If True, internal storage is (n_markers, n_individuals)
                       for fast individual subsetting on memmaps
        """
        self._transposed = transposed

        if isinstance(data, np.memmap):
            self._data = data
            self._is_memmap = True
        elif isinstance(data, np.ndarray):
            self._data = data
            self._is_memmap = False
        elif isinstance(data, (str, Path)):
            # Memory-mapped file
            if shape is None:
                raise ValueError("Shape required for memory-mapped files")
            self._data = np.memmap(data, dtype=dtype, mode='r', shape=shape)
            self._is_memmap = True
        else:
            raise ValueError("Data must be array or file path")

        base_n_individuals = self._data.shape[1] if self._transposed else self._data.shape[0]
        if row_indexer is None:
            self._row_indexer = None
        else:
            normalized_row_indexer = np.asarray(row_indexer, dtype=np.int64)
            if normalized_row_indexer.ndim != 1:
                raise ValueError("row_indexer must be a 1D array")
            if normalized_row_indexer.size:
                if normalized_row_indexer.min() < 0 or normalized_row_indexer.max() >= base_n_individuals:
                    raise IndexError("row_indexer is out of bounds for genotype matrix")
                if np.array_equal(normalized_row_indexer, np.arange(base_n_individuals, dtype=np.int64)):
                    normalized_row_indexer = None
            self._row_indexer = normalized_row_indexer

        # Track if data has been pre-imputed (no -9 values)
        # This allows downstream code to skip -9 checks for faster processing
        self._is_imputed = is_imputed

        # Pre-compute major alleles for efficient imputation
        self._major_alleles = None
        self._missing_masks = None
        # Skip precompute when data is already imputed (no -9 values).
        if precompute_alleles and not self._is_imputed:
            self._precompute_major_alleles()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (n_individuals, n_markers)"""
        n_individuals = len(self._row_indexer) if self._row_indexer is not None else (
            self._data.shape[1] if self._transposed else self._data.shape[0]
        )
        n_markers = self._data.shape[0] if self._transposed else self._data.shape[1]
        return (n_individuals, n_markers)

    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return self.shape[0]

    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return self.shape[1]

    @property
    def is_imputed(self) -> bool:
        """Whether missing values (-9) have been pre-imputed.

        When True, downstream code can skip -9 checks for faster processing.
        """
        return self._is_imputed

    @property
    def is_transposed(self) -> bool:
        """Whether internal storage is transposed (n_markers, n_individuals).

        When True, individual subsetting is fast (column slice on memmap).
        """
        return self._transposed

    @property
    def is_memmap(self) -> bool:
        """Whether the underlying storage is a numpy memmap."""
        return self._is_memmap

    @property
    def estimated_nbytes(self) -> int:
        """Estimated byte size of the matrix in API row-major shape."""
        return int(self.n_individuals * self.n_markers * self._data.dtype.itemsize)

    @property
    def has_row_subset(self) -> bool:
        """Whether this matrix lazily views a subset of individuals."""
        return self._row_indexer is not None

    def _normalize_subset_indices(self, indices: Union[np.ndarray, list]) -> np.ndarray:
        if isinstance(indices, list):
            indices = np.asarray(indices)
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            if indices.ndim != 1 or indices.size != self.n_individuals:
                raise ValueError("Boolean individual indexer must match the number of individuals")
            return np.flatnonzero(indices).astype(np.int64, copy=False)

        normalized = np.asarray(indices, dtype=np.int64)
        if normalized.ndim != 1:
            raise ValueError("Individual indices must be a 1D array-like")
        if normalized.size:
            if normalized.min() < 0 or normalized.max() >= self.n_individuals:
                raise IndexError("Individual indices are out of bounds")
        return normalized

    def _compose_row_indexer(self, indices: np.ndarray) -> np.ndarray:
        if self._row_indexer is None:
            return indices.astype(np.int64, copy=False)
        return self._row_indexer[indices]

    def _materialize_rows(self, row_indexer: np.ndarray) -> np.ndarray:
        """Copy a row subset into a standalone ndarray in API row-major order."""
        row_indexer = np.asarray(row_indexer, dtype=np.int64)
        if row_indexer.ndim != 1:
            raise ValueError("row_indexer must be a 1D array")
        if self._transposed:
            return np.array(self._data[:, row_indexer].T, copy=True)
        if row_indexer.size == 0:
            return np.empty((0, self.n_markers), dtype=self._data.dtype)

        # Fast path for memmaps: when rows are monotonic and mostly contiguous
        # (common after genotype-order alignment), copy long row runs directly.
        if self._is_memmap and row_indexer.size > 1:
            diffs = np.diff(row_indexer)
            if np.all(diffs >= 0):
                run_starts = np.concatenate(([0], np.flatnonzero(diffs != 1) + 1))
                run_ends = np.concatenate((run_starts[1:], [row_indexer.size]))
                n_runs = int(run_starts.size)
                max_runs_for_chunk_copy = min(2048, max(32, row_indexer.size // 4))
                if n_runs <= max_runs_for_chunk_copy:
                    out = np.empty((row_indexer.size, self.n_markers), dtype=self._data.dtype)
                    for start, end in zip(run_starts, run_ends):
                        src_start = int(row_indexer[start])
                        src_end = int(row_indexer[end - 1]) + 1
                        out[start:end, :] = self._data[src_start:src_end, :]
                    return out
                out = np.empty((row_indexer.size, self.n_markers), dtype=self._data.dtype)
                np.take(self._data, row_indexer, axis=0, out=out)
                return out

        return np.array(self._data[row_indexer, :], copy=True)

    @staticmethod
    def _as_contiguous_marker_slice(
        indices: np.ndarray,
    ) -> Optional[slice]:
        """Return a slice when marker indices form one contiguous block."""
        marker_idx = np.asarray(indices, dtype=np.int64)
        if marker_idx.ndim != 1 or marker_idx.size == 0:
            return None
        if marker_idx.size == 1:
            start = int(marker_idx[0])
            return slice(start, start + 1)
        if np.all(np.diff(marker_idx) == 1):
            return slice(int(marker_idx[0]), int(marker_idx[-1]) + 1)
        return None

    def _select_marker_block(
        self,
        markers: Union[slice, np.ndarray],
    ) -> np.ndarray:
        if self._transposed:
            if self._row_indexer is None:
                return self._data[markers, :].T
            if isinstance(markers, slice):
                return self._data[markers, :][:, self._row_indexer].T
            marker_idx = np.asarray(markers, dtype=np.int64)
            return self._data[np.ix_(marker_idx, self._row_indexer)].T

        if self._row_indexer is None:
            return self._data[:, markers]
        if isinstance(markers, slice):
            # For row-major memmaps, slice markers first so numpy can use a
            # contiguous view, then gather rows from the smaller block.
            block = self._data[:, markers]
            return block[self._row_indexer, :]
        marker_idx = np.asarray(markers, dtype=np.int64)
        return self._data[np.ix_(self._row_indexer, marker_idx)]

    def get_columns(
        self,
        indices: Union[np.ndarray, list],
        *,
        dtype: Optional[np.dtype] = None,
        copy: bool = False,
    ) -> np.ndarray:
        """Get arbitrary marker columns without altering missing values."""
        if isinstance(indices, list):
            indices = np.asarray(indices, dtype=np.int64)
        marker_idx = np.asarray(indices, dtype=np.int64)
        marker_slice = self._as_contiguous_marker_slice(marker_idx)
        marker_block = self._select_marker_block(marker_slice if marker_slice is not None else marker_idx)
        if dtype is not None:
            return np.asarray(marker_block, dtype=np.dtype(dtype))
        if copy:
            return np.array(marker_block, copy=True)
        return np.asarray(marker_block)

    def __getitem__(self, key):
        """Support array indexing - returns data in (individuals, markers) order"""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(col_key, (int, np.integer)):
                return self.get_marker(int(col_key))[row_key]
            if isinstance(col_key, slice):
                start, stop, step = col_key.indices(self.n_markers)
                batch = self.get_batch(start, stop)
                return batch[row_key, ::step]
            columns = np.arange(self.n_markers, dtype=np.int64)[col_key]
            return self.get_columns(columns)[row_key]

        if isinstance(key, (int, np.integer)):
            return self.get_individual(int(key))

        row_indices = np.arange(self.n_individuals, dtype=np.int64)[key]
        return self.subset_individuals(row_indices).to_numpy(copy=False)

    def get_marker(self, marker_idx: int) -> np.ndarray:
        """Get genotypes for a specific marker"""
        if self._row_indexer is None:
            if self._transposed:
                return self._data[marker_idx, :]
            return self._data[:, marker_idx]
        if self._transposed:
            return np.asarray(self._data[marker_idx, self._row_indexer])
        column = self._data[:, marker_idx]
        return np.asarray(column[self._row_indexer])

    def get_individual(self, ind_idx: int) -> np.ndarray:
        """Get genotypes for a specific individual"""
        base_idx = int(self._row_indexer[ind_idx]) if self._row_indexer is not None else int(ind_idx)
        if self._transposed:
            return self._data[:, base_idx]
        return self._data[base_idx, :]

    def get_batch(self, marker_start: int, marker_end: int) -> np.ndarray:
        """Get batch of markers for efficient processing.

        Returns array of shape (n_individuals, n_markers_in_batch).
        """
        return np.asarray(self._select_marker_block(slice(marker_start, marker_end)))

    def subset_individuals(
        self,
        indices: Union[np.ndarray, list],
        *,
        precompute_alleles: Optional[bool] = None,
        materialize: bool = False,
    ) -> "GenotypeMatrix":
        """Return a GenotypeMatrix restricted to a subset of individuals.

        Preserves the is_imputed flag so downstream code can use fast paths
        when genotypes are already imputed.

        When ``materialize=False`` (default), this returns a lazy row view.
        When ``materialize=True``, it copies the subset into a standalone
        row-major ndarray, which can still be useful for access patterns that
        repeatedly touch many scattered markers.
        """
        indexer = self._normalize_subset_indices(indices)
        composed_row_indexer = self._compose_row_indexer(indexer)

        if precompute_alleles is None:
            precompute_alleles = not self._is_imputed
        if materialize:
            subset = self._materialize_rows(composed_row_indexer)
            return GenotypeMatrix(
                subset,
                precompute_alleles=precompute_alleles,
                is_imputed=self._is_imputed,
                transposed=False,
            )
        return GenotypeMatrix(
            self._data,
            precompute_alleles=precompute_alleles,
            is_imputed=self._is_imputed,
            transposed=self._transposed,
            row_indexer=composed_row_indexer,
        )

    def subset_markers(
        self,
        indices: Union[np.ndarray, list],
        *,
        precompute_alleles: Optional[bool] = None,
    ) -> "GenotypeMatrix":
        """Return a GenotypeMatrix restricted to a subset of markers.

        Always materializes the selected columns into a standalone (n_ind,
        n_markers_kept) array. Supports boolean masks or integer index arrays.
        """
        if isinstance(indices, list):
            indices = np.asarray(indices)
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            if indices.ndim != 1 or indices.size != self.n_markers:
                raise ValueError("Boolean marker indexer must match the number of markers")
            marker_idx = np.flatnonzero(indices).astype(np.int64, copy=False)
        else:
            marker_idx = np.asarray(indices, dtype=np.int64)
            if marker_idx.ndim != 1:
                raise ValueError("Marker indices must be a 1D array-like")
            if marker_idx.size:
                if marker_idx.min() < 0 or marker_idx.max() >= self.n_markers:
                    raise IndexError("Marker indices are out of bounds")

        if marker_idx.size == 0:
            subset = np.empty((self.n_individuals, 0), dtype=self._data.dtype)
        else:
            subset = np.ascontiguousarray(self.get_columns(marker_idx))

        if precompute_alleles is None:
            precompute_alleles = not self._is_imputed
        return GenotypeMatrix(
            subset,
            precompute_alleles=precompute_alleles,
            is_imputed=self._is_imputed,
            transposed=False,
        )

    def calculate_allele_frequencies(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate allele frequencies for all markers.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        n_markers = self.n_markers
        frequencies = np.zeros(n_markers)

        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)
            # Frequency of alt allele = mean(genotype) / max_dosage
            frequencies[start:end] = np.mean(batch, axis=0) / max(max_dosage, 1e-12)

        return frequencies

    def calculate_maf(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate minor allele frequencies.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        frequencies = self.calculate_allele_frequencies(
            batch_size=batch_size,
            max_dosage=max_dosage,
        )
        return np.minimum(frequencies, 1 - frequencies)
    
    def _precompute_major_alleles(self, batch_size: int = 1000):
        """Pre-compute major alleles for all markers to optimize missing data imputation
        
        This matches rMVP's missing data imputation strategy exactly.
        """
        n_markers = self.n_markers
        self._major_alleles = np.zeros(n_markers, dtype=self._data.dtype)
        
        # Process in batches to handle large datasets
        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)

            if batch.size == 0:
                continue

            # Missing mask aligns with rMVP sentinel handling
            missing_mask = (batch == -9) | np.isnan(batch)
            non_missing_counts = (~missing_mask).sum(axis=0)
            completely_missing = non_missing_counts == 0

            if np.all(completely_missing):
                # All markers in this block are entirely missing
                self._major_alleles[start:end] = 0
                continue

            valid_values = batch[~missing_mask]
            if valid_values.size == 0:
                self._major_alleles[start:end] = 0
                continue

            unique_vals = np.unique(valid_values)
            if unique_vals.size == 0:
                self._major_alleles[start:end] = 0
                continue

            unique_vals = unique_vals.astype(self._data.dtype, copy=False)
            counts = np.zeros((unique_vals.size, end - start), dtype=np.int32)
            for idx, val in enumerate(unique_vals):
                counts[idx, :] = np.sum(batch == val, axis=0)

            major_indices = np.argmax(counts, axis=0)
            major_vals = unique_vals[major_indices]
            if np.any(completely_missing):
                major_vals = major_vals.astype(self._data.dtype, copy=False)
                major_vals[completely_missing] = 0
            self._major_alleles[start:end] = major_vals
    
    def get_marker_imputed(self,
                           marker_idx: int,
                           *,
                           fill_value: Optional[float] = None,
                           dtype: np.dtype = np.float64) -> np.ndarray:
        """Get genotypes for a specific marker with optional missing-data imputation."""
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: data contains no missing values, so skip mask checks.
            return self.get_marker(marker_idx).astype(out_dtype, copy=True)

        marker = self.get_marker(marker_idx).astype(out_dtype, copy=True)

        missing_mask = (marker == -9) | np.isnan(marker)
        if not missing_mask.any():
            return marker

        if fill_value is not None:
            marker[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            marker[missing_mask] = out_dtype.type(self._major_alleles[marker_idx])
        else:
            marker[missing_mask] = out_dtype.type(0.0)

        return marker

    def get_batch_imputed(self,
                          marker_start: int,
                          marker_end: int,
                          *,
                          fill_value: Optional[float] = None,
                          dtype: np.dtype = np.float64) -> np.ndarray:
        """Get batch of markers with missing data imputed.

        Args:
            marker_start: Inclusive start index
            marker_end: Exclusive end index
            fill_value: Optional constant to impute missing genotypes.
                If None, the pre-computed major allele is used (rMVP default).
            dtype: Output dtype for the returned array (default float64).

        Returns:
            Array of shape (n_individuals, n_markers_in_batch).
        """
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: pre-imputed data, no missing checks needed.
            return self.get_batch(marker_start, marker_end).astype(out_dtype, copy=True)

        batch = self.get_batch(marker_start, marker_end).astype(out_dtype, copy=True)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[marker_start:marker_end].astype(out_dtype, copy=False)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = out_dtype.type(0.0)

        return batch

    def get_columns_imputed(self,
                             indices: Union[np.ndarray, list],
                             *,
                             fill_value: Optional[float] = None,
                             dtype: np.dtype = np.float64) -> np.ndarray:
        """Get arbitrary marker columns with missing data imputed.

        Optimized for fetching non-contiguous markers (e.g., pseudo-QTNs) in one call.
        Returns an array of shape (n_individuals, len(indices)) with requested dtype.
        """
        if isinstance(indices, list):
            indices = np.array(indices, dtype=int)
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: pre-imputed data, no missing checks needed.
            return self.get_columns(indices, dtype=out_dtype)

        # Slice and copy to ensure we don't mutate underlying storage
        batch = self.get_columns(indices, dtype=out_dtype)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[indices].astype(out_dtype, copy=False)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = out_dtype.type(0.0)

        return batch
    
    @property
    def major_alleles(self) -> Optional[np.ndarray]:
        """Pre-computed major alleles for all markers"""
        return self._major_alleles

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """Materialize the genotype matrix as a numpy array in API row order."""
        if not copy and self._row_indexer is None and not self._transposed:
            return np.asarray(self._data)
        array = self.get_batch(0, self.n_markers)
        if copy:
            return np.array(array, copy=True)
        return np.asarray(array)


def ensure_eager_genotype(
    geno: Union[GenotypeMatrix, np.ndarray],
) -> Union[GenotypeMatrix, np.ndarray]:
    """Materialize lazy genotype row subsets for scan-heavy GWAS methods."""
    if isinstance(geno, GenotypeMatrix) and geno.has_row_subset:
        return geno.subset_individuals(
            np.arange(geno.n_individuals, dtype=np.int64),
            materialize=True,
            precompute_alleles=not geno.is_imputed,
        )
    return geno


class AssociationResults:
    """GWAS association results structure
    
    Standard format: [Effect, SE, P-value] for each marker
    """
    
    def __init__(self, effects: np.ndarray, se: np.ndarray, pvalues: np.ndarray,
                 snp_map: Optional[GenotypeMap] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        # Validate lengths (first dimension must match)
        n = len(effects)
        if hasattr(se, '__len__') and len(se) != n:
             raise ValueError(f"SE array length {len(se)} does not match effects length {n}")
        if hasattr(pvalues, '__len__') and len(pvalues) != n:
             raise ValueError(f"P-value array length {len(pvalues)} does not match effects length {n}")
        if snp_map is not None:
            if hasattr(snp_map, "n_markers"):
                map_n = int(snp_map.n_markers)
            elif hasattr(snp_map, "to_dataframe"):
                map_n = len(snp_map.to_dataframe())
            else:
                map_n = len(snp_map)
            if map_n != n:
                raise ValueError(
                    f"AssociationResults marker count {n} does not match SNP map length {map_n}"
                )

        self.effects = effects
        self.se = se  
        self.pvalues = pvalues
        self.snp_map = snp_map
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self.effects)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'Effect': self.effects,
            'SE': self.se,
            'P-value': self.pvalues
        })
        
        if self.snp_map is not None:
            marker_values = self.snp_map.marker_ids.values
            df[MARKER_ID_COLUMN] = marker_values
            # Keep legacy column for backward compatibility.
            df[LEGACY_MARKER_ID_COLUMN] = marker_values
            df['Chr'] = self.snp_map.chromosomes.values  
            df['Pos'] = self.snp_map.positions.values
            ordered_cols = [
                MARKER_ID_COLUMN,
                LEGACY_MARKER_ID_COLUMN,
                'Chr',
                'Pos',
                'Effect',
                'SE',
                'P-value',
            ]
            df = df[ordered_cols]
            
        return df
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [Effect, SE, P-value]
        
        If results contain multiple columns (e.g. for covariates),
        only the first column (marker results) is returned by default
        to maintain backward compatibility with 2D array expectations.
        """
        eff = self.effects if self.effects.ndim == 1 else self.effects[:, 0]
        se = self.se if self.se.ndim == 1 else self.se[:, 0]
        pv = self.pvalues if self.pvalues.ndim == 1 else self.pvalues[:, 0]
        
        return np.column_stack([eff, se, pv])

    def manhattan_plot(self,
                       map_data: Optional[GenotypeMap] = None,
                       threshold: float = 5e-8,
                       title: str = "Manhattan Plot",
                       figsize: Tuple[int, int] = (12, 6),
                       output_file: Optional[Union[str, Path]] = None):
        """Create a Manhattan plot directly from this result object."""
        from ..visualization.manhattan import create_manhattan_plot

        effective_map = map_data if map_data is not None else self.snp_map
        fig = create_manhattan_plot(
            pvalues=self.pvalues if self.pvalues.ndim == 1 else self.pvalues[:, 0],
            map_data=effective_map,
            threshold=threshold,
            title=title,
            figsize=figsize,
        )
        if output_file is not None:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
        return fig

    def qq_plot(self,
                title: str = "Q-Q Plot",
                figsize: Tuple[int, int] = (6, 6),
                output_file: Optional[Union[str, Path]] = None):
        """Create a QQ plot directly from this result object."""
        from ..visualization.manhattan import create_qq_plot

        fig = create_qq_plot(
            pvalues=self.pvalues if self.pvalues.ndim == 1 else self.pvalues[:, 0],
            title=title,
            figsize=figsize,
        )
        if output_file is not None:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
        return fig


class KinshipMatrix:
    """Kinship matrix with validation and properties
    
    Must be symmetric positive semi-definite matrix
    """
    
    def __init__(self, data: Union[np.ndarray, str, Path]):
        if isinstance(data, (str, Path)):
            # Load from file - handle CSV with headers
            try:
                df = pd.read_csv(data, header=0)
                self._data = df.values.astype(float)
            except:
                # Try without headers  
                self._data = np.loadtxt(data, delimiter=',', skiprows=1)
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            raise ValueError("Data must be array or file path")
            
        # Validate properties
        if self._data.ndim != 2:
            raise ValueError("Kinship matrix must be 2D")
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Kinship matrix must be square")
        if not np.allclose(self._data, self._data.T, atol=1e-10):
            raise ValueError("Kinship matrix must be symmetric")
            
        self.n = self._data.shape[0]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape"""
        return self._data.shape
    
    def __getitem__(self, key):
        """Support array indexing"""
        return self._data[key]
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data.copy()
    
    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigendecomposition for MLM"""
        eigenvals, eigenvecs = np.linalg.eigh(self._data)
        return eigenvals, eigenvecs


def load_validation_data(data_path: Union[str, Path]) -> Dict[str, Any]:
    """Load validation data for testing
    
    Returns dictionary with test data matching R rMVP outputs
    """
    data_path = Path(data_path)
    
    result = {}
    
    # Load test data
    if (data_path / "test_phenotype.csv").exists():
        result['phenotype'] = Phenotype(data_path / "test_phenotype.csv")
    
    if (data_path / "test_genotype_full.csv").exists():
        geno_data = pd.read_csv(data_path / "test_genotype_full.csv").values
        result['genotype'] = GenotypeMatrix(geno_data)
    
    if (data_path / "test_map.csv").exists():
        result['map'] = GenotypeMap(data_path / "test_map.csv")
    
    # Load expected results
    if (data_path / "test_glm_results.csv").exists():
        glm_data = pd.read_csv(data_path / "test_glm_results.csv").values
        result['expected_glm'] = AssociationResults(
            glm_data[:, 0], glm_data[:, 1], glm_data[:, 2]
        )
    
    if (data_path / "test_mlm_results.csv").exists():
        mlm_data = pd.read_csv(data_path / "test_mlm_results.csv").values  
        result['expected_mlm'] = AssociationResults(
            mlm_data[:, 0], mlm_data[:, 1], mlm_data[:, 2]
        )
    
    if (data_path / "test_kinship.csv").exists():
        result['expected_kinship'] = KinshipMatrix(data_path / "test_kinship.csv")
        
    if (data_path / "test_pca_results.csv").exists():
        result['expected_pca'] = pd.read_csv(data_path / "test_pca_results.csv").values
    
    return result
