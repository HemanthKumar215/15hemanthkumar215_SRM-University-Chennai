"""
src/ingestion – Ingestion, Validation & Harmonization for Uploaded Data
======================================================================
"""

from src.ingestion.parsers import (
    BaseCGMParser,
    FreeStyleLibreParser,
    DexcomParser,
    AppleHealthParser,
    CanonicalParser,
    detect_and_parse_cgm,
)
from src.ingestion.validator import (
    ValidationResult,
    validate_cgm_data,
    validate_ehr_data,
    validate_patient_data,
    COHORT_MEDIANS,
)

__all__ = [
    "BaseCGMParser",
    "FreeStyleLibreParser",
    "DexcomParser",
    "AppleHealthParser",
    "CanonicalParser",
    "detect_and_parse_cgm",
    "ValidationResult",
    "validate_cgm_data",
    "validate_ehr_data",
    "validate_patient_data",
    "COHORT_MEDIANS",
]
