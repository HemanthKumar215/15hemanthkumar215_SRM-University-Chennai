# 📚 Real-World Diabetes Datasets & Adapters

This document details the real-world clinical datasets supported by the Digital Twin framework, including data source links, licensing, schema mappings, and instructions for ingestion.

> ⚠️ **Data Privacy & Governance**:
> - Never commit raw or de-identified real patient datasets to git.
> - Ensure compliance with applicable regulations: **DPDP Act 2023** (India), **HIPAA** (USA), and **GDPR** (EU).
> - All raw real-world data directories are ignored via `.gitignore` under `data/real/`.

---

## 1. Supported Datasets

### A. AI-READI (Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights)
- **Source**: [https://aireadi.org](https://aireadi.org) / NIH Bridge2AI Consortium
- **Cohort**: Multi-ethnic cohort spanning Healthy, Prediabetes, and Type 2 Diabetes participants.
- **Sensors & Modalities**:
  - Continuous Glucose Monitoring (Dexcom G6, 5-min intervals)
  - Consumer Wearables (Garmin vívoactive 4S: HR, steps, sleep staging, stress)
  - Clinical EHR (demographics, labs, HbA1c, fasting lipids, retinal imaging)
- **License**: Bridge2AI Open-Access Research DUA (Data Use Agreement)
- **Access Steps**:
  1. Register at [Fairhub / AI-READI Data Portal](https://fairhub.io/studies/aireadi).
  2. Complete the Data Use Agreement (DUA) and human subjects ethics training acknowledgment.
  3. Download the wearable and CGM tabular files (`cgm_readings.csv`, `garmin_metrics.csv`, `participant_ehr.csv`).
  4. Place in `data/real/aireadi/`.

---

### B. CGMacros Benchmark Dataset
- **Source**: [https://github.com/irinagain/cgm-data](https://github.com/irinagain/cgm-data) / PhysioNet
- **Cohort**: Standardized benchmark of CGM recordings from diverse clinical research trials (Type 1, Type 2, and non-diabetic controls).
- **Sensors & Modalities**:
  - High-resolution CGM (5-minute and 15-minute sampling rates)
  - Clinical metadata (age, sex, diabetes type, baseline HbA1c)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Access Steps**:
  1. Clone or download from repository or PhysioNet archive.
  2. Place CSV files (e.g., `cgmacros_cgm.csv`, `cgmacros_participants.csv`) in `data/real/cgmacros/`.

---

### C. Shanghai T1DM / T2DM Dataset
- **Source**: [Nature Scientific Data / Figshare](https://doi.org/10.1038/s41597-023-01940-7)
- **Cohort**: 34 Type 2 Diabetes and 12 Type 1 Diabetes inpatients and outpatients monitored over 14 days by the Shanghai Clinical Center for Diabetes.
- **Sensors & Modalities**:
  - Abbott Freestyle Libre CGM (15-min intervals, resampled to 5-min intervals by the adapter)
  - Detailed daily meal logs (carbohydrate grams, meal timings, breakfast/lunch/dinner)
  - Basic clinical profiles (age, sex, BMI, disease duration)
- **License**: Creative Commons Attribution 4.0 (CC-BY 4.0)
- **Access Steps**:
  1. Download the Figshare repository archive containing `Patient_*.xlsx` or `Patient_*.csv`.
  2. Place in `data/real/shanghai/` (e.g. `data/real/shanghai/cgm/` and `data/real/shanghai/meals/`).

---

## 2. Common Standardized Schema

All adapters in `src/data_real/` ingest heterogeneous vendor and study formats and map them into the following unified schema:

### Dynamic Time-Series Schema (`cgm_wearable`)
| Column Name | Data Type | Units / Range | Description |
|---|---|---|---|
| `patient_id` | `str` | e.g., `SH-001`, `AR-101` | Unique subject identifier |
| `timestamp` | `datetime64[ns]` | ISO 8601 UTC/Local | 5-minute uniform grid timestamp |
| `glucose_mg_dl` | `float32` | $40 - 450$ mg/dL | Calibrated interstitial glucose |
| `heart_rate_bpm` | `float32` | $40 - 200$ bpm | Wearable photoplethysmography (PPG) HR |
| `hrv_ms` | `float32` | $5 - 150$ ms | RMSSD metric of heart rate variability |
| `steps` | `int32` | $\ge 0$ | Cumulative steps in 5-min epoch |
| `sleep_stage` | `int32` | $0=\text{Awake}, 1=\text{Light}, 2=\text{Deep}, 3=\text{REM}$ | Hypnogram classification |
| `meal_carbs` | `int32` | grams ($0 - 200$) | Logged carbohydrate intake |
| `meal_type` | `str` | `breakfast`, `lunch`, etc. | Dietary event annotation |
| `exercise_intensity`| `float32` | $0.0 - 1.0$ | Relative exertion intensity |
| `exercise_duration` | `int32` | minutes | Reported duration of physical exertion |

### Static EHR Schema (`ehr_profiles`)
| Column Name | Data Type | Units / Values | Description |
|---|---|---|---|
| `patient_id` | `str` | Unique ID | Primary key |
| `age` | `int32` | $18 - 90$ years | Chronological age |
| `sex` | `str` | `'M'`, `'F'` | Biological sex |
| `bmi` | `float32` | $16 - 55\text{ kg/m}^2$ | Body mass index |
| `hba1c` | `float32` | $4.0 - 16.0\%$ | Glycated hemoglobin |
| `fasting_glucose` | `float32` | mg/dL | Baseline fasting venous plasma glucose |
| `sbp` / `dbp` | `int32` | mmHg | Systolic and diastolic blood pressure |
| `cohort` | `str` | `Normal`, `Prediabetic`, `T2D` | Clinical diagnostic category |

---

## 3. Handling Missing Modalities (Real Data Adaptation)

Real-world datasets rarely provide every sensor simultaneously. For example:
- **Shanghai**: Offers dense CGM + meal logs, but no continuous PPG heart rate or sleep tracking.
- **CGMacros**: Pure CGM with sparse clinical covariates.
- **AI-READI**: Full multimodal wearable and clinical observations.

### Modality Masking & Forward Imputation Strategy:
1. **Modality Mask Matrix**: Each batch includes a binary mask vector $M \in \{0, 1\}^K$ indicating which channels are recorded for the given patient.
2. **Missing Input Handling**: Unmeasured sensor channels are filled with population median values and marked as masked in the attention/GRU heads.
3. **Pretraining with Modality Dropout**: During Stage A pretraining on synthetic data, channels are randomly dropped with probability $p=0.3$ so the multi-branch network becomes resilient to missing modalities when evaluated on real-world cohorts.
