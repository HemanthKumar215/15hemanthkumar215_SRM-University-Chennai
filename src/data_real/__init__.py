"""
src/data_real – Real-world Diabetes Dataset Adapters & Loaders
==============================================================
Provides adapters for AI-READI, CGMacros, and Shanghai T1DM/T2DM datasets,
mapping external sensor and EHR formats into the canonical digital twin schema.
"""

from src.data_real.base import (
    BaseRealAdapter,
    CANONICAL_CGM_COLS,
    CANONICAL_EHR_COLS,
)
from src.data_real.aireadi_adapter import AIREADIAdapter
from src.data_real.cgmacros_adapter import CGMacrosAdapter
from src.data_real.shanghai_adapter import ShanghaiAdapter
from src.data_real.mock_generator import generate_all_mock_datasets

__all__ = [
    "BaseRealAdapter",
    "CANONICAL_CGM_COLS",
    "CANONICAL_EHR_COLS",
    "AIREADIAdapter",
    "CGMacrosAdapter",
    "ShanghaiAdapter",
    "generate_all_mock_datasets",
]
