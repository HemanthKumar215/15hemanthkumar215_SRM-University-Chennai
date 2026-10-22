"""
ehr_generator.py – Synthetic Electronic Health Record generator
================================================================
Generates static patient profiles for a South-Asian-focused T2D cohort.

All data is **100 % synthetic**.  No real patient data is used.
Parameters are drawn from published epidemiological distributions for
India / South Asia (ICMR-INDIAB, Joshi et al. 2015, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Indian first-name pools (common, gender-appropriate) for realistic patient IDs
# ──────────────────────────────────────────────────────────────────────────────
_MALE_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Rohan", "Karthik", "Rahul", "Vikram",
    "Suresh", "Rajesh", "Manoj", "Amit", "Deepak", "Sanjay",
]
_FEMALE_NAMES = [
    "Saanvi", "Aanya", "Aadhya", "Diya", "Ananya", "Isha", "Priya",
    "Pooja", "Meera", "Kavya", "Lakshmi", "Sneha", "Neha", "Divya",
    "Anjali", "Sunita", "Rekha", "Suman", "Geeta", "Radha",
]

# Medication lists
_T2D_MEDICATIONS = [
    "metformin", "sulfonylurea", "insulin", "dpp4_inhibitor", "sglt2_inhibitor",
]
_OTHER_MEDICATIONS = ["antihypertensive", "statin"]


def _sample_age(rng: np.random.Generator, n: int) -> np.ndarray:
    """Age distribution skewed toward 40-60 (peak T2D onset in South Asia)."""
    ages = rng.normal(loc=50, scale=12, size=n).clip(25, 75).astype(int)
    return ages


def _sample_bmi(rng: np.random.Generator, n: int, ages: np.ndarray,
                sexes: np.ndarray) -> np.ndarray:
    """BMI – South Asians have higher metabolic risk at lower BMI (>23).
    Women tend slightly higher BMI post-menopause."""
    base = rng.normal(loc=26.5, scale=4.5, size=n)
    # slight increase with age
    base += (ages - 40) * 0.04
    # females slightly higher after 45
    female_mask = (sexes == "F") & (ages > 45)
    base[female_mask] += rng.normal(1.0, 0.5, size=female_mask.sum())
    return base.clip(18.5, 42.0).round(1)


def _sample_hba1c(
    rng: np.random.Generator,
    n: int,
    cohort: np.ndarray,   # 0=normal, 1=prediabetic, 2=t2d
    bmis: np.ndarray,
) -> np.ndarray:
    """HbA1c calibrated to produce 3 distinct clinical categories.

    * normal (0): HbA1c 4.0–5.6  (mean 5.1 ± 0.35)
    * prediabetic (1): HbA1c 5.7–6.4 (mean 6.1 ± 0.25)
    * T2D (2): HbA1c 6.5–13 (mean 8.0 ± 1.4, right-skewed)
    """
    hba1c = np.empty(n)
    norm_mask = cohort == 0
    pre_mask = cohort == 1
    t2d_mask = cohort == 2

    hba1c[norm_mask] = rng.normal(5.1, 0.35, int(norm_mask.sum())).clip(4.0, 5.6)
    hba1c[pre_mask] = rng.normal(6.1, 0.25, int(pre_mask.sum())).clip(5.7, 6.4)
    # T2D: use a log-normal-like shape to give right tail (HbA1c 9-13)
    t2d_raw = rng.normal(8.0, 1.4, int(t2d_mask.sum()))
    hba1c[t2d_mask] = t2d_raw.clip(6.5, 13.0)

    # BMI adds a small bump (insulin resistance)
    hba1c += np.maximum(bmis - 25.0, 0) * 0.015
    return hba1c.clip(4.0, 13.0).round(1)


def generate_ehr_profiles(
    n_patients: int | None = None,
    seed: int | None = None,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate a DataFrame of synthetic EHR profiles.

    Parameters
    ----------
    n_patients : int, optional
        Number of patients (default from config).
    seed : int, optional
        Random seed (default from config).
    config : dict, optional
        Full config dict; loaded automatically if not given.

    Returns
    -------
    pd.DataFrame
        One row per patient with static clinical features.
    """
    if config is None:
        config = load_config()
    if n_patients is None:
        n_patients = config["data"]["n_patients"]
    if seed is None:
        seed = config["project"]["seed"]

    rng = np.random.default_rng(seed)
    demo = config["demographics"]

    logger.info("Generating %d synthetic EHR profiles (seed=%d)…", n_patients, seed)

    # --- Demographics ---
    sexes = np.where(rng.random(n_patients) < demo["male_fraction"], "M", "F")
    ages = _sample_age(rng, n_patients)
    ethnicities = np.where(
        rng.random(n_patients) < demo["south_asian_fraction"],
        "South Asian", "Other"
    )

    # --- Diagnoses: strict 30/30/40 cohort mix ---
    # 0 = normal, 1 = prediabetic, 2 = T2D
    cohort_probs = [0.30, 0.30, 0.40]
    cohort = rng.choice([0, 1, 2], size=n_patients, p=cohort_probs)
    has_t2d = (cohort == 2)
    has_prediabetes = (cohort == 1)
    has_htn = rng.random(n_patients) < demo["hypertension_prevalence"]
    has_dyslip = rng.random(n_patients) < demo["dyslipidemia_prevalence"]
    # T2D patients have higher HTN co-occurrence (~55%)
    has_htn[has_t2d] = rng.random(int(has_t2d.sum())) < 0.55
    # Prediabetic: moderate HTN risk (~30%)
    has_htn[has_prediabetes] = rng.random(int(has_prediabetes.sum())) < 0.30

    # --- Anthropometrics & labs ---
    bmis = _sample_bmi(rng, n_patients, ages, sexes)
    hba1c = _sample_hba1c(rng, n_patients, cohort, bmis)

    # Fasting glucose derived from Nathan formula with ±8 mg/dL noise
    # 28.7 * HbA1c - 46.7 gives the estimated average glucose; fasting is
    # ~70-80% of the daily mean for non-diabetics, ~85% for diabetics.
    eag = 28.7 * hba1c - 46.7
    fasting_fraction = np.where(has_t2d, 0.82, 0.75)
    fasting_glucose = (eag * fasting_fraction + rng.normal(0, 8, n_patients)).clip(65, 300).round(0)

    # Lipid profile
    total_chol = rng.normal(200, 35, n_patients).clip(120, 350).round(0)
    ldl = rng.normal(120, 30, n_patients).clip(50, 250).round(0)
    hdl_base = np.where(sexes == "F", 55, 45)
    hdl = (hdl_base + rng.normal(0, 10, n_patients)).clip(25, 100).round(0)
    triglycerides = rng.normal(160, 60, n_patients).clip(50, 500).round(0)

    # Blood pressure
    systolic = rng.normal(130, 18, n_patients).clip(90, 200).round(0)
    systolic[has_htn] += rng.normal(12, 5, has_htn.sum())
    systolic = systolic.clip(90, 200).round(0)
    diastolic = (systolic * 0.6 + rng.normal(0, 5, n_patients)).clip(55, 120).round(0)

    # --- Family history & genetic risk ---
    family_history_dm = rng.random(n_patients) < 0.45  # high in South Asians
    family_history_cvd = rng.random(n_patients) < 0.25
    # Genetic risk score (0-1 scale, higher = more risk)
    grs = rng.beta(2, 3, n_patients).round(3)
    # South Asians get a bump
    grs[ethnicities == "South Asian"] += 0.1
    grs = grs.clip(0, 1)

    # --- Medications (T2D only) ---
    med_metformin = has_t2d & (rng.random(n_patients) < 0.80)
    med_sulfonylurea = has_t2d & (rng.random(n_patients) < 0.35)
    # Insulin prescribed for poorly-controlled T2D (HbA1c > 8.5) and some prediabetics
    med_insulin = has_t2d & (hba1c > 8.5) & (rng.random(n_patients) < 0.50)
    med_dpp4i = has_t2d & (rng.random(n_patients) < 0.25)
    med_sglt2i = has_t2d & (rng.random(n_patients) < 0.20)
    med_antihtn = has_htn & (rng.random(n_patients) < 0.70)
    med_statin = has_dyslip & (rng.random(n_patients) < 0.60)

    # --- Lifestyle ---
    smoking = rng.choice(["never", "former", "current"], n_patients,
                         p=[0.60, 0.25, 0.15])
    activity_level = rng.choice(["sedentary", "light", "moderate", "active"],
                                n_patients, p=[0.30, 0.35, 0.25, 0.10])

    # --- HbA1c history (last 4 quarterly values) ---
    hba1c_history: List[str] = []
    for i in range(n_patients):
        hist = [round(hba1c[i] + rng.normal(0, 0.3), 1) for _ in range(4)]
        hist.sort()  # oldest to newest trend
        if rng.random() < 0.4:
            hist.reverse()  # worsening trend for some
        hba1c_history.append(";".join(str(h) for h in hist))

    # --- Assemble DataFrame ---
    patient_ids = [f"P{i:04d}" for i in range(n_patients)]
    names = []
    for i in range(n_patients):
        pool = _MALE_NAMES if sexes[i] == "M" else _FEMALE_NAMES
        names.append(rng.choice(pool) + f" {rng.choice(['Kumar','Singh','Sharma','Reddy','Patel','Nair','Iyer','Das','Gupta','Rao'])}")

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "name": names,
        "age": ages,
        "sex": sexes,
        "ethnicity": ethnicities,
        "bmi": bmis,
        "hba1c": hba1c,
        "cohort": np.where(has_t2d, "T2D", np.where(has_prediabetes, "prediabetic", "normal")),
        "hba1c_history": hba1c_history,
        "fasting_glucose": fasting_glucose.astype(int),
        "total_cholesterol": total_chol.astype(int),
        "ldl": ldl.astype(int),
        "hdl": hdl.astype(int),
        "triglycerides": triglycerides.astype(int),
        "systolic_bp": systolic.astype(int),
        "diastolic_bp": diastolic.astype(int),
        "diagnosed_t2d": has_t2d.astype(int),
        "diagnosed_hypertension": has_htn.astype(int),
        "diagnosed_dyslipidemia": has_dyslip.astype(int),
        "medication_metformin": med_metformin.astype(int),
        "medication_sulfonylurea": med_sulfonylurea.astype(int),
        "medication_insulin": med_insulin.astype(int),
        "medication_dpp4i": med_dpp4i.astype(int),
        "medication_sglt2i": med_sglt2i.astype(int),
        "medication_antihypertensive": med_antihtn.astype(int),
        "medication_statin": med_statin.astype(int),
        "family_history_diabetes": family_history_dm.astype(int),
        "family_history_cvd": family_history_cvd.astype(int),
        "genetic_risk_score": grs.round(3),
        "smoking_status": smoking,
        "physical_activity_level": activity_level,
    })

    logger.info("EHR generation complete.  Shape: %s", df.shape)
    return df
