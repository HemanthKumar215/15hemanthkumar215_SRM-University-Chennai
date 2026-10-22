"""
cgm_generator.py – Synthetic CGM and wearable time-series generator
=====================================================================
Produces realistic continuous glucose monitor, heart rate, HRV, step count,
and sleep stage data for each patient over ``n_days`` at 5-minute resolution.

Key physiological features simulated
--------------------------------------
* **Mean glucose calibrated to Nathan formula**: mean ≈ 28.7*HbA1c - 46.7 ± 10 mg/dL
* **Fasting baseline** anchored to EHR ``fasting_glucose``
* **Spike amplitude** scales with HbA1c, BMI, insulin resistance, and carb load
* **HbA1c-gated upper bound**: HbA1c < 5.7 → never exceeds ~160 mg/dL;
  HbA1c > 9 → can reach 300–400 mg/dL
* **Circadian rhythm** in basal glucose
* **Dawn phenomenon** (early-morning hepatic glucose output)
* **Meal-driven excursions** with pharmacokinetic absorption curves;
  carb loads and timing follow an Indian dietary pattern;
  random timing ±45 min, variable carb loads, ~15% meal-skip probability
* **Exercise-induced glucose lowering**
* **Poor-sleep effect** on next-day insulin resistance
* **Medication effects** (metformin, sulfonylurea, insulin, DPP-4i, SGLT2-i)
* **Sensor noise** and **missing values**
* **Inter-day variability**: random baseline drift so no two days look identical

Cohort mix (controlled by EHR generator):
  ~30% normal (HbA1c 4.0–5.6), ~30% prediabetic (5.7–6.4), ~40% T2D (≥6.5)

All data is **100 % synthetic**.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Minutes in a day / readings per day at 5-min intervals
_MIN_PER_DAY = 1440
_READINGS_PER_DAY = 288  # 1440 / 5


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _estimated_avg_glucose(hba1c: float) -> float:
    """Nathan formula: convert HbA1c → estimated average glucose (mg/dL)."""
    return 28.7 * hba1c - 46.7


def _soft_limit_glucose(
    glucose: np.ndarray,
    soft_thresh: float = 340.0,
    max_bound: float = 400.0,
) -> np.ndarray:
    """Smooth physiological soft-limit towards 400 mg/dL (never a flat plateau).

    Uses a smooth tanh asymptotic curve above soft_thresh so the derivative remains
    strictly positive everywhere, preventing unnatural horizontal clipping.
    """
    scale = max_bound - soft_thresh
    excess = np.maximum(glucose - soft_thresh, 0.0)
    saturated = soft_thresh + scale * np.tanh(excess / scale)
    out = np.where(glucose > soft_thresh, saturated, glucose)
    return np.maximum(out, 40.0)


def _insulin_resistance_factor(hba1c: float, bmi: float) -> float:
    """Dimensionless multiplier (≥ 1.0) reflecting insulin resistance."""
    ir = 1.0
    ir += max(0.0, (hba1c - 5.7)) * 0.18   # +0.18 per unit above 5.7
    ir += max(0.0, (bmi - 23.0)) * 0.025    # +0.025 per BMI unit above 23
    return min(ir, 3.5)                      # cap to prevent runaway


def _medication_modifier(row: pd.Series) -> float:
    """Glucose-lowering modifier (≤ 1.0). Smaller → more lowering."""
    mod = 1.0
    if row.get("medication_metformin", 0):
        mod *= 0.88
    if row.get("medication_sulfonylurea", 0):
        mod *= 0.90
    if row.get("medication_insulin", 0):
        mod *= 0.80
    if row.get("medication_dpp4i", 0):
        mod *= 0.93
    if row.get("medication_sglt2i", 0):
        mod *= 0.91
    return mod


def _circadian(hour: np.ndarray, amplitude: float = 8.0) -> np.ndarray:
    """Circadian basal glucose oscillation – trough ~03:00, peak ~18:00."""
    return amplitude * np.sin(2 * np.pi * (hour - 3) / 24)


def _dawn_effect(hour: np.ndarray, magnitude: float = 18.0) -> np.ndarray:
    """Dawn phenomenon: hepatic glucose output ramps 04:00–08:00."""
    center, width = 6.0, 1.5
    return magnitude * np.exp(-0.5 * ((hour - center) / width) ** 2)


def _meal_glucose_response(
    minutes_since_meal: np.ndarray,
    carbs: float,
    ir_factor: float,
    t_peak: float = 45.0,
    peak_amp_factor: float = 1.0,
    alpha: float = 2.0,
) -> np.ndarray:
    """Glucose excursion from a single meal using a smooth gamma-absorption curve.

    The curve rises smoothly over 30–60 min (cresting at t_peak), with per-meal variation
    in peak height and timing, and decays gradually over 2.5–4 hours without abrupt plateaus.

    Formula:
        f(t) = A * (t / t_peak)^alpha * exp(-alpha * (t / t_peak - 1))
    """
    t = np.maximum(minutes_since_meal, 0.0).astype(float)
    # Peak amplitude proportional to carbs & insulin resistance, scaled by per-meal factor
    carb_norm = carbs / 45.0
    # Healthy individuals secrete insulin rapidly, keeping meal spikes modest (~25-35 mg/dL)
    insulin_sensitivity_attenuation = 0.65 if ir_factor <= 1.05 else (0.85 if ir_factor <= 1.25 else 1.0)
    amp = 36.0 * carb_norm * ir_factor * peak_amp_factor * insulin_sensitivity_attenuation

    # Gamma response (peak exactly at t_peak with derivative 0)
    ratio = t / max(t_peak, 10.0)
    curve = (ratio ** alpha) * np.exp(-alpha * (ratio - 1.0))
    # Zero before meal
    curve = np.where(minutes_since_meal >= 0.0, curve, 0.0)
    return amp * curve


def _exercise_effect(
    minutes_since_start: np.ndarray,
    duration_min: float,
    intensity: float,
    ir_factor: float,
) -> np.ndarray:
    """Glucose-lowering effect of exercise."""
    t = np.maximum(minutes_since_start, 0).astype(float)
    during = (t <= duration_min).astype(float)
    after = (t > duration_min).astype(float)
    time_after = np.maximum(t - duration_min, 0)
    post_decay = np.exp(-time_after / 120.0)
    drop = intensity * 25.0 / ir_factor
    return -drop * (during + after * post_decay)


def _generate_meal_schedule(
    rng: np.random.Generator,
    n_days: int,
    config: Dict[str, Any],
    meal_skip_prob: float = 0.15,
) -> List[List[Dict[str, Any]]]:
    """Return a list of daily meal lists; each meal is {time_hour, carbs, name}.

    Adds ±45-min jitter to every meal and per-day carb variation of ±20 %.
    """
    templates = config["meals"]["templates"]
    schedule: List[List[Dict[str, Any]]] = []

    for _ in range(n_days):
        day_meals: List[Dict[str, Any]] = []
        for tmpl in templates:
            # Skip the occasional meal (realistic for any meal, not just late_snack)
            base_skip = 1.0 - tmpl.get("probability", 1.0)
            effective_skip = max(base_skip, meal_skip_prob)
            if rng.random() < effective_skip:
                continue

            t_lo, t_hi = tmpl["time_range"]
            carb_lo, carb_hi = tmpl["carb_range"]

            # ±45-min jitter on meal timing
            raw_time = rng.uniform(t_lo, t_hi)
            jitter_h = rng.uniform(-0.75, 0.75)
            meal_time = float(np.clip(raw_time + jitter_h, t_lo - 0.5, t_hi + 0.5))

            # ±20% variability on carb load
            base_carbs = int(rng.integers(carb_lo, carb_hi + 1))
            factor = rng.uniform(0.80, 1.20)
            carbs = int(np.clip(base_carbs * factor, carb_lo * 0.7, carb_hi * 1.3))

            meal = {
                "name": tmpl["name"],
                "food": rng.choice(tmpl["options"]),
                "time_hour": round(meal_time, 2),
                "carbs": carbs,
            }
            day_meals.append(meal)
        schedule.append(day_meals)
    return schedule


def _generate_exercise_events(
    rng: np.random.Generator,
    n_days: int,
    activity_level: str,
) -> List[List[Dict[str, Any]]]:
    """Generate daily exercise events based on patient activity level."""
    prob_map = {"sedentary": 0.10, "light": 0.30, "moderate": 0.55, "active": 0.80}
    p = prob_map.get(activity_level, 0.30)

    events: List[List[Dict[str, Any]]] = []
    for _ in range(n_days):
        day_ex: List[Dict[str, Any]] = []
        if rng.random() < p:
            t = rng.choice([rng.uniform(6.0, 8.0), rng.uniform(17.0, 19.5)])
            dur = int(rng.integers(20, 60))
            intensity = float(rng.uniform(0.3, 0.9))
            if activity_level == "sedentary":
                intensity *= 0.6
            day_ex.append({
                "time_hour": round(float(t), 2),
                "duration_min": dur,
                "intensity": round(intensity, 2),
            })
        events.append(day_ex)
    return events


def _generate_sleep_quality(
    rng: np.random.Generator,
    n_days: int,
) -> np.ndarray:
    """Sleep quality score per night (0-1, 1 = excellent)."""
    base = rng.beta(5, 2, n_days)
    bad_nights = rng.random(n_days) < 0.15
    base[bad_nights] *= rng.uniform(0.3, 0.6, int(bad_nights.sum()))
    return base.round(2)


def _step_count_profile(
    hour: np.ndarray,
    activity_level: str,
    exercise_events: List[Dict[str, Any]],
    age: int,
    bmi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Steps per 5-minute interval calibrated to realistic daily totals (2,000–12,000 steps/day).

    Diurnal pattern features morning/lunch/evening active periods, exercise bursts,
    and near-zero activity during sleep.
    """
    # Daily target by activity level
    target_map = {
        "sedentary": (2500, 4200),
        "light": (4500, 6800),
        "moderate": (7200, 9500),
        "active": (9800, 12500),
    }
    lo, hi = target_map.get(activity_level, (4500, 6800))
    daily_target = float(rng.uniform(lo, hi))

    # Adjust for age and BMI
    age_factor = max(0.78, 1.0 - max(0, age - 50) * 0.005)
    bmi_factor = max(0.78, 1.0 - max(0, bmi - 27.0) * 0.008)
    daily_target *= age_factor * bmi_factor

    awake = (hour >= 6.0) & (hour < 23.0)

    # Base diurnal intensity weights
    diurnal = np.zeros(len(hour))
    # Morning commute / walk
    diurnal += 2.2 * np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
    # Midday activity
    diurnal += 1.8 * np.exp(-0.5 * ((hour - 13.0) / 1.2) ** 2)
    # Evening activity
    diurnal += 2.5 * np.exp(-0.5 * ((hour - 18.5) / 1.5) ** 2)
    # Baseline awake walking
    diurnal += 0.8 * awake.astype(float)

    # Exercise bursts
    for ex in exercise_events:
        t_start = ex["time_hour"]
        dur_h = ex["duration_min"] / 60.0
        mask = (hour >= t_start) & (hour < t_start + dur_h)
        diurnal[mask] += 6.5 * ex["intensity"]

    # Sleep hours: minimal movement (rare shift in bed)
    night = (hour < 6.0) | (hour >= 23.0)
    diurnal[night] = 0.015 * rng.random(int(night.sum()))

    # Normalize weights to daily_target
    tot_wt = diurnal.sum()
    if tot_wt > 0:
        expected_steps = (diurnal / tot_wt) * daily_target
    else:
        expected_steps = np.full(len(hour), daily_target / len(hour))

    steps = rng.poisson(np.maximum(expected_steps, 0.0))
    steps[night & (rng.random(len(hour)) > 0.05)] = 0
    return steps


def _heart_rate_profile(
    hour: np.ndarray,
    age: int,
    bmi: float,
    exercise_events: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Heart rate (bpm) with circadian variation and exercise peaks."""
    resting_hr = 60 + (age - 30) * 0.15 + (bmi - 23) * 0.4 + rng.normal(0, 3)
    resting_hr = float(np.clip(resting_hr, 55, 100))
    circadian_hr = 8 * np.sin(2 * np.pi * (hour - 4) / 24)
    hr = resting_hr + circadian_hr + rng.normal(0, 3, len(hour))
    for ex in exercise_events:
        t_start = ex["time_hour"]
        dur_h = ex["duration_min"] / 60
        mask = (hour >= t_start) & (hour < t_start + dur_h)
        hr[mask] += 30 * ex["intensity"] + rng.normal(0, 5, int(mask.sum()))
        post = (hour >= t_start + dur_h) & (hour < t_start + dur_h + 0.5)
        hr[post] += 10 * ex["intensity"]
    sleep_mask = (hour < 6) | (hour > 23)
    hr[sleep_mask] -= 10
    return hr.clip(45, 185).round(0)


def _hrv_profile(
    hour: np.ndarray,
    age: int,
    hr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """HRV (RMSSD, ms) – inversely related to HR, lower with age."""
    base_hrv = max(20, 80 - (age - 25) * 0.8)
    hrv = base_hrv * (70.0 / np.maximum(hr, 50)) + rng.normal(0, 5, len(hour))
    sleep_mask = (hour < 6) | (hour > 23)
    hrv[sleep_mask] += 15
    return hrv.clip(5, 150).round(1)


def _sleep_stages(
    hour: np.ndarray,
    quality: float,
    age: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full physiological sleep hypnogram (5–9 h per night) with Light/Deep/REM cycles and brief awakenings.

    Encoding:
    0 = Awake
    1 = Light Sleep (N1 / N2)
    2 = Deep Sleep (N3 / SWS)
    3 = REM Sleep
    """
    stages = np.zeros(len(hour), dtype=int)

    # 5.2 to 8.8 hours of sleep per night modulated by quality & age
    base_dur = rng.normal(7.3, 0.7)
    duration_h = float(np.clip(base_dur * (0.85 + 0.25 * quality), 5.2, 8.8))

    bedtime = float(rng.uniform(22.0, 23.6))
    wake_time = (bedtime + duration_h) % 24.0

    if bedtime > wake_time:
        in_bed = (hour >= bedtime) | (hour < wake_time)
    else:
        in_bed = (hour >= bedtime) & (hour < wake_time)

    if not in_bed.any():
        return stages

    in_bed_idx = np.where(in_bed)[0]
    total_in_bed = len(in_bed_idx)

    # ~90-minute sleep cycles (18 intervals of 5-min)
    cycle_len = int(rng.integers(16, 21))

    for step_num, idx in enumerate(in_bed_idx):
        cycle_phase = (step_num % cycle_len) / float(cycle_len)
        progress = step_num / float(max(1, total_in_bed))

        # Sleep latency (first 10–15 min light or awake)
        if step_num < 3:
            stages[idx] = 1 if rng.random() < 0.65 else 0
            continue

        if cycle_phase < 0.22:
            stages[idx] = 1  # Light
        elif cycle_phase < 0.62:
            # Deep sleep predominant in first third of night
            deep_p = max(0.08, 0.85 - 0.80 * progress)
            stages[idx] = 2 if rng.random() < deep_p else 1
        elif cycle_phase < 0.94:
            # REM sleep expands in second half of night
            rem_p = min(0.85, 0.20 + 0.70 * progress)
            stages[idx] = 3 if rng.random() < rem_p else 1
        else:
            stages[idx] = 1

    # Brief awakenings (WASO - Wake After Sleep Onset): 1 to 4 brief awakenings
    n_wakes = int(rng.integers(1, 5 if quality < 0.5 else 3))
    if total_in_bed > 25:
        centers = rng.choice(in_bed_idx[4:-4], size=min(n_wakes, total_in_bed - 10), replace=False)
        for c in centers:
            dur = rng.choice([1, 2])
            stages[c: min(c + dur, len(stages))] = 0

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_patient_timeseries(
    patient_row: pd.Series,
    n_days: int = 30,
    interval_min: int = 5,
    config: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate multi-channel time-series for a single patient.

    The glucose trace is **calibrated to the patient's EHR**:

    * Basal level set so that the 30-day mean approximates
      ``28.7 * HbA1c - 46.7`` (Nathan formula) ± 10 mg/dL.
    * Fasting (overnight, 02:00–05:00) glucose matches ``fasting_glucose``
      from the EHR.
    * Spike amplitude and ceiling are capped per HbA1c category so normal
      patients never reach diabetic levels.

    Parameters
    ----------
    patient_row : pd.Series
        One row from the EHR profiles DataFrame.
    n_days, interval_min : int
    config : dict, optional
    seed : int, optional

    Returns
    -------
    pd.DataFrame
        Columns: patient_id, timestamp, day, hour, glucose_mg_dl,
        heart_rate_bpm, hrv_ms, steps, sleep_stage, meal_event, meal_carbs,
        exercise_event, exercise_intensity.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = config["project"]["seed"]

    pid = patient_row["patient_id"]
    patient_seed = seed + hash(pid) % (2 ** 31)
    rng = np.random.default_rng(patient_seed & 0x7FFFFFFF)

    # ── Patient physiological parameters ──────────────────────────────────
    hba1c = float(patient_row["hba1c"])
    bmi = float(patient_row["bmi"])
    age = int(patient_row["age"])
    fasting_glucose_ehr = float(patient_row["fasting_glucose"])

    eag = _estimated_avg_glucose(hba1c)                   # Nathan formula target
    ir_factor = _insulin_resistance_factor(hba1c, bmi)
    med_mod = _medication_modifier(patient_row)
    activity_level = str(patient_row.get("physical_activity_level", "light"))

    # ── Pre-generate schedules ─────────────────────────────────────────────
    meal_schedule = _generate_meal_schedule(rng, n_days, config)
    exercise_schedule = _generate_exercise_events(rng, n_days, activity_level)
    sleep_quality = _generate_sleep_quality(rng, n_days)

    readings_per_day = _MIN_PER_DAY // interval_min
    total_readings = n_days * readings_per_day

    day_idx = np.repeat(np.arange(n_days), readings_per_day)
    minute_of_day = np.tile(np.arange(0, _MIN_PER_DAY, interval_min), n_days)
    hour_of_day = minute_of_day / 60.0

    # ── Glucose simulation ──────────────────────────────────────────────────
    # Start from fasting glucose as the overnight baseline (02:00-05:00 anchor)
    # and calibrate so the overall mean hits eag after meal responses.
    estimated_meal_offset = min(45.0, 32.0 * ir_factor * med_mod)
    basal_target = max(fasting_glucose_ehr, eag * med_mod - estimated_meal_offset)
    # Per-patient random offset within ±8 mg/dL for natural variability
    basal_jitter = float(rng.uniform(-8.0, 8.0))
    basal = float(np.clip(basal_target + basal_jitter, 60.0, 240.0))

    glucose = np.full(total_readings, basal)

    # Circadian + dawn (scale with IR factor)
    circ_amp = min(8.0 * ir_factor, 18.0)
    dawn_mag = min(12.0 * ir_factor * med_mod, 35.0)
    glucose += _circadian(hour_of_day, amplitude=circ_amp)
    glucose += _dawn_effect(hour_of_day, magnitude=dawn_mag)

    # ── Per-day baseline drift so no two days look the same ───────────────
    day_drift = rng.normal(0.0, 5.0, n_days)
    glucose += np.repeat(day_drift, readings_per_day)

    # ── Meal responses ─────────────────────────────────────────────────────
    meal_events = np.zeros(total_readings, dtype=int)
    meal_carbs_arr = np.zeros(total_readings)
    for d in range(n_days):
        day_start = d * readings_per_day
        for meal in meal_schedule[d]:
            meal_time_min = meal["time_hour"] * 60
            meal_idx = day_start + int(round(meal_time_min / interval_min))
            if 0 <= meal_idx < total_readings:
                meal_events[meal_idx] = 1
                meal_carbs_arr[meal_idx] = meal["carbs"]
            day_minutes = minute_of_day[day_start: day_start + readings_per_day]
            mins_since = day_minutes - meal_time_min

            # Per-meal variability in absorption peak time (30-60 min) and amplitude
            t_peak = float(np.clip(rng.uniform(36.0, 50.0) * (1.0 + 0.10 * (ir_factor - 1.0)), 30.0, 60.0))
            amp_fac = float(rng.uniform(0.88, 1.15))
            alpha_val = float(rng.uniform(1.85, 2.25))

            response = _meal_glucose_response(
                mins_since, meal["carbs"], ir_factor,
                t_peak=t_peak, peak_amp_factor=amp_fac, alpha=alpha_val
            )
            glucose[day_start: day_start + readings_per_day] += response

    # ── Exercise effects ───────────────────────────────────────────────────
    ex_events = np.zeros(total_readings, dtype=int)
    ex_intensity = np.zeros(total_readings)
    for d in range(n_days):
        day_start = d * readings_per_day
        for ex in exercise_schedule[d]:
            ex_time_min = ex["time_hour"] * 60
            ex_idx = day_start + int(round(ex_time_min / interval_min))
            if 0 <= ex_idx < total_readings:
                ex_events[ex_idx] = 1
                ex_intensity[ex_idx] = ex["intensity"]
            day_minutes = minute_of_day[day_start: day_start + readings_per_day]
            mins_since = day_minutes - ex_time_min
            glucose[day_start: day_start + readings_per_day] += _exercise_effect(
                mins_since, ex["duration_min"], ex["intensity"], ir_factor
            )

    # ── Poor-sleep next-day glucose bump ──────────────────────────────────
    for d in range(1, n_days):
        if sleep_quality[d - 1] < 0.4:
            bump = (0.4 - sleep_quality[d - 1]) * 18.0
            day_start = d * readings_per_day
            glucose[day_start: day_start + readings_per_day] += bump

    # ── Intra-day random walk (slow drift, zero-mean per day) ─────────────
    drift = np.cumsum(rng.normal(0, 0.22, total_readings))
    drift -= np.repeat(
        [drift[i * readings_per_day: (i + 1) * readings_per_day].mean()
         for i in range(n_days)],
        readings_per_day,
    )
    glucose += drift

    # ── Sensor noise ──────────────────────────────────────────────────────
    noise_std = float(config["data"]["noise_std_glucose"])
    glucose += rng.normal(0, noise_std, total_readings)

    # ── Physiological soft-limit (never hard flat clamp) ──────────────────
    if hba1c < 5.7:
        s_th, m_bd = 145.0, 162.0
    elif hba1c < 6.5:
        s_th, m_bd = 185.0, 210.0
    else:
        s_th, m_bd = 340.0, 400.0

    glucose = _soft_limit_glucose(glucose, soft_thresh=s_th, max_bound=m_bd).round(1)

    # ── Calibration correction: nudge mean to match Nathan formula ────────
    if n_days >= 5:
        fasting_mask = (hour_of_day >= 2.0) & (hour_of_day < 5.0)
        current_fasting_mean = float(np.nanmean(glucose[fasting_mask]))
        fasting_correction = float(
            np.clip(fasting_glucose_ehr - current_fasting_mean, -25.0, 25.0)
        )
        glucose += fasting_correction * 0.35
        glucose = _soft_limit_glucose(glucose, soft_thresh=s_th, max_bound=m_bd).round(1)

    # ── Missing values ─────────────────────────────────────────────────────
    missing_rate = float(config["data"]["missing_rate"])
    missing_mask = rng.random(total_readings) < missing_rate
    # Occasional longer gap (sensor removal / recharge: 1-2 h)
    if rng.random() < 0.20:
        gap_start = int(rng.integers(0, total_readings - 25))
        gap_len = int(rng.integers(6, 24))
        missing_mask[gap_start: gap_start + gap_len] = True
    glucose[missing_mask] = np.nan

    # ── Other channels ─────────────────────────────────────────────────────
    all_hr = np.empty(total_readings)
    all_hrv = np.empty(total_readings)
    all_steps = np.empty(total_readings)
    all_sleep = np.empty(total_readings, dtype=int)

    for d in range(n_days):
        s, e = d * readings_per_day, (d + 1) * readings_per_day
        day_hour = hour_of_day[s:e]
        ex_day = exercise_schedule[d]
        hr = _heart_rate_profile(day_hour, age, bmi, ex_day, rng)
        all_hr[s:e] = hr
        all_hrv[s:e] = _hrv_profile(day_hour, age, hr, rng)
        all_steps[s:e] = _step_count_profile(day_hour, activity_level, ex_day, age, bmi, rng)
        all_sleep[s:e] = _sleep_stages(day_hour, sleep_quality[d], age, rng)

    hr_noise_std = float(config["data"]["noise_std_hr"])
    all_hr += rng.normal(0, hr_noise_std, total_readings)
    all_hr = all_hr.clip(40, 200).round(0)

    # ── Build timestamps ──────────────────────────────────────────────────
    base_ts = pd.Timestamp("2026-01-01")
    timestamps = base_ts + pd.to_timedelta(
        day_idx * _MIN_PER_DAY + minute_of_day, unit="min"
    )

    df = pd.DataFrame({
        "patient_id": pid,
        "timestamp": timestamps,
        "day": day_idx,
        "hour": hour_of_day.round(4),
        "glucose_mg_dl": glucose,
        "heart_rate_bpm": all_hr.astype(int),
        "hrv_ms": all_hrv.round(1),
        "steps": all_steps.astype(int),
        "sleep_stage": all_sleep,
        "meal_event": meal_events,
        "meal_carbs": meal_carbs_arr.astype(int),
        "exercise_event": ex_events,
        "exercise_intensity": ex_intensity.round(2),
    })

    return df


def generate_all_timeseries(
    ehr_df: pd.DataFrame,
    n_days: int | None = None,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate CGM/wearable time-series for all patients in ``ehr_df``."""
    if config is None:
        config = load_config()
    if n_days is None:
        n_days = config["data"]["n_days"]

    frames: List[pd.DataFrame] = []
    n = len(ehr_df)
    for i, (_, row) in enumerate(ehr_df.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Generating time-series for patient %d / %d …", i + 1, n)
        frames.append(generate_patient_timeseries(row, n_days=n_days, config=config))

    logger.info("Concatenating %d patient time-series …", n)
    full_df = pd.concat(frames, ignore_index=True)
    logger.info("CGM/wearable data shape: %s", full_df.shape)
    return full_df
