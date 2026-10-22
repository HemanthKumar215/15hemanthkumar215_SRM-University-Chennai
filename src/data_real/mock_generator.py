"""
mock_generator.py – Generate Mock Real Data for CI / Testing
=============================================================
Creates realistic mock directories for AI-READI, CGMacros, and Shanghai
matching raw study schemas, allowing immediate end-to-end testing of the
real-data adapters and fine-tuning pipeline without downloading restricted files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get_project_root

logger = logging.getLogger(__name__)


def generate_mock_aireadi(output_dir: Path, n_patients: int = 15, n_days: int = 14) -> None:
    """Generate mock raw AI-READI files: cgm_readings.csv, garmin_metrics.csv, participant_ehr.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)

    pids = [f"AR_{i:03d}" for i in range(1, n_patients + 1)]
    cohorts = rng.choice(["Control", "Prediabetes", "Type 2 Diabetes"], size=n_patients, p=[0.3, 0.3, 0.4])

    # Participant EHR
    ehr_records = []
    for pid, cohort in zip(pids, cohorts):
        hba1c = round(float(rng.uniform(5.0, 5.6) if cohort == "Control"
                            else rng.uniform(5.8, 6.4) if cohort == "Prediabetes"
                            else rng.uniform(6.8, 10.5)), 1)
        fasting_glu = round(float(28.7 * hba1c - 46.7 + rng.normal(0, 8)), 1)
        ehr_records.append({
            "study_id": pid,
            "age": int(rng.integers(30, 75)),
            "sex": rng.choice(["M", "F"]),
            "bmi": round(float(rng.uniform(21.0, 36.0)), 1),
            "glycated_hemoglobin": hba1c,
            "fasting_plasma_glucose": max(65.0, fasting_glu),
            "systolic_bp": int(rng.integers(110, 155)),
            "diastolic_bp": int(rng.integers(70, 95)),
            "diabetes_status": cohort,
        })
    pd.DataFrame(ehr_records).to_csv(output_dir / "participant_ehr.csv", index=False)

    # CGM Readings & Garmin Wearable (5-min intervals)
    cgm_records = []
    garmin_records = []
    base_ts = pd.Timestamp("2026-03-01 00:00:00")
    total_steps = n_days * 288

    for pid, cohort in zip(pids, cohorts):
        hba1c = [r["glycated_hemoglobin"] for r in ehr_records if r["study_id"] == pid][0]
        mean_glu = 28.7 * hba1c - 46.7
        # Random walk with mean reversion
        glu_series = np.empty(total_steps)
        glu_series[0] = mean_glu
        for t in range(1, total_steps):
            glu_series[t] = glu_series[t - 1] + 0.15 * (mean_glu - glu_series[t - 1]) + rng.normal(0, 4)
            # Add occasional meal spikes
            if (t % 288) in (96, 156, 240) and rng.random() < 0.85:
                glu_series[t] += rng.uniform(25, 75)
        glu_series = np.clip(glu_series, 45, 380).round(1)

        timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(total_steps)]

        for ts, glu in zip(timestamps, glu_series):
            cgm_records.append({"study_id": pid, "datetime": ts, "cgm_value": glu})
            garmin_records.append({
                "study_id": pid,
                "datetime": ts,
                "hr": int(rng.integers(60, 110)),
                "rmssd": round(float(rng.uniform(15, 65)), 1),
                "step_count": int(rng.poisson(12)),
                "sleep": int(rng.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])),
            })

    pd.DataFrame(cgm_records).to_csv(output_dir / "cgm_readings.csv", index=False)
    pd.DataFrame(garmin_records).to_csv(output_dir / "garmin_metrics.csv", index=False)
    logger.info("Generated mock AI-READI files in: %s", output_dir)


def generate_mock_cgmacros(output_dir: Path, n_patients: int = 15, n_days: int = 14) -> None:
    """Generate mock raw CGMacros files: cgmacros_cgm.csv, cgmacros_participants.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(202)

    pids = [f"CGM_{i:03d}" for i in range(1, n_patients + 1)]
    cohorts = rng.choice(["Control", "T2D"], size=n_patients, p=[0.4, 0.6])

    ehr_records = []
    for pid, cohort in zip(pids, cohorts):
        hba1c = round(float(rng.uniform(5.0, 5.6) if cohort == "Control"
                            else rng.uniform(6.8, 10.2)), 1)
        ehr_records.append({
            "subject_id": pid,
            "age": int(rng.integers(28, 70)),
            "gender": rng.choice(["M", "F"]),
            "bmi": round(float(rng.uniform(22.0, 35.0)), 1),
            "hba1c": hba1c,
            "diabetes_type": cohort,
        })
    pd.DataFrame(ehr_records).to_csv(output_dir / "cgmacros_participants.csv", index=False)

    cgm_records = []
    base_ts = pd.Timestamp("2026-03-01 00:00:00")
    total_steps = n_days * 288

    for pid in pids:
        hba1c = [r["hba1c"] for r in ehr_records if r["subject_id"] == pid][0]
        mean_glu = 28.7 * hba1c - 46.7
        glu_series = np.empty(total_steps)
        glu_series[0] = mean_glu
        for t in range(1, total_steps):
            glu_series[t] = glu_series[t - 1] + 0.15 * (mean_glu - glu_series[t - 1]) + rng.normal(0, 4)
            if (t % 288) in (90, 160, 245) and rng.random() < 0.8:
                glu_series[t] += rng.uniform(20, 70)
        glu_series = np.clip(glu_series, 45, 380).round(1)

        timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(total_steps)]
        for ts, glu in zip(timestamps, glu_series):
            cgm_records.append({"subject_id": pid, "time": ts, "glucose": glu})

    pd.DataFrame(cgm_records).to_csv(output_dir / "cgmacros_cgm.csv", index=False)
    logger.info("Generated mock CGMacros files in: %s", output_dir)


def generate_mock_shanghai(output_dir: Path, n_patients: int = 15, n_days: int = 14) -> None:
    """Generate mock raw Shanghai files: shanghai_cgm.csv, shanghai_meals.csv, shanghai_patients.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(303)

    pids = [f"SH_{i:03d}" for i in range(1, n_patients + 1)]
    ehr_records = []
    for pid in pids:
        hba1c = round(float(rng.uniform(6.5, 9.8)), 1)
        ehr_records.append({
            "subject_id": pid,
            "age": int(rng.integers(40, 75)),
            "gender": rng.choice(["M", "F"]),
            "bmi": round(float(rng.uniform(23.0, 32.0)), 1),
            "hba1c": hba1c,
            "diabetes_type": "T2D",
        })
    pd.DataFrame(ehr_records).to_csv(output_dir / "shanghai_patients.csv", index=False)

    # 15-minute intervals for Shanghai Libre CGM
    cgm_records = []
    meal_records = []
    base_ts = pd.Timestamp("2026-03-01 00:00:00")
    steps_15m = n_days * 96

    for pid in pids:
        hba1c = [r["hba1c"] for r in ehr_records if r["subject_id"] == pid][0]
        mean_glu = 28.7 * hba1c - 46.7
        glu_series = np.empty(steps_15m)
        glu_series[0] = mean_glu
        for t in range(1, steps_15m):
            glu_series[t] = glu_series[t - 1] + 0.20 * (mean_glu - glu_series[t - 1]) + rng.normal(0, 5)
            if (t % 96) in (32, 52, 80) and rng.random() < 0.9:
                carbs = int(rng.integers(40, 95))
                glu_series[t] += carbs * 0.7
                meal_records.append({
                    "subject_id": pid,
                    "datetime": base_ts + pd.Timedelta(minutes=15 * t),
                    "carbs_g": carbs,
                })
        glu_series = np.clip(glu_series, 45, 380).round(1)

        timestamps = [base_ts + pd.Timedelta(minutes=15 * i) for i in range(steps_15m)]
        for ts, glu in zip(timestamps, glu_series):
            cgm_records.append({"subject_id": pid, "datetime": ts, "glucose": glu})

    pd.DataFrame(cgm_records).to_csv(output_dir / "shanghai_cgm.csv", index=False)
    pd.DataFrame(meal_records).to_csv(output_dir / "shanghai_meals.csv", index=False)
    logger.info("Generated mock Shanghai files in: %s", output_dir)


def generate_all_mock_datasets(base_real_dir: Optional[Path] = None) -> None:
    """Generate all three mock datasets inside base_real_dir (default: data/real/)."""
    if base_real_dir is None:
        base_real_dir = get_project_root() / "data" / "real"
    base_real_dir.mkdir(parents=True, exist_ok=True)
    generate_mock_aireadi(base_real_dir / "aireadi")
    generate_mock_cgmacros(base_real_dir / "cgmacros")
    generate_mock_shanghai(base_real_dir / "shanghai")
    print(f"[OK] Generated mock real-world datasets in: {base_real_dir}")


if __name__ == "__main__":
    generate_all_mock_datasets()
