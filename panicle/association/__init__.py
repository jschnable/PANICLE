"""
Association testing methods for GWAS analysis
"""

from .glm import PANICLE_GLM, PANICLE_GLM_MULTI
from .mlm import PANICLE_MLM
from .farmcpu import PANICLE_FarmCPU
from .blink import PANICLE_BLINK
from .bayes_loco import PANICLE_BayesLOCO, BayesLocoConfig

__all__ = [
    'PANICLE_GLM',
    'PANICLE_GLM_MULTI',
    'PANICLE_MLM',
    'PANICLE_FarmCPU',
    'PANICLE_BLINK',
    'PANICLE_BayesLOCO',
    'BayesLocoConfig',
]
