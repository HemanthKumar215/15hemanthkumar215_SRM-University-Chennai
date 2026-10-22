"""
generate_all.py – Scalable Synthetic EHR + CGM Data Generator
==============================================================
Run with:
    python -m src.data_gen.generate_all [--cohort_size 500] [--days 30] [--partitioned]

Supports scaling to 10,000+ patients using partitioned Parquet streaming,
ensuring low memory consumption on laptops.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import get_project_root, load_config
from src.utils.logging_setup import setup_logging
from src.data_gen.ehr_generator import generate_ehr_profiles
from src.data_gen.cgm_generator import generate_patient_timeseries

logger = logging.getLogger(__name__)


def generate_partitioned_cgm(
    ehr_df: pd.DataFrame,
    output_dir: Path,
    config: dict,
    chunk_size: int = 100,
    n_days: int = 30,
) -> Path:
    """Generate CGM records in chunks and write directly to partitioned Parquet files."""
    partitions_dir = output_dir / "cgm_partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)

    n_patients = len(ehr_df)
    total_chunks = (n_patients + chunk_size - 1) // chunk_size

    logger.info("Streaming CGM generation into %d chunks (chunk_size=%d)...", total_chunks, chunk_size)

    for chunk_idx in range(total_chunks):
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, n_patients)
        sub_ehr = ehr_df.iloc[start_i:end_i]

        chunk_frames: List[pd.DataFrame] = []
        for _, patient_row in sub_ehr.iterrows():
            ts = generate_patient_timeseries(patient_row, n_days=n_days, config=config)
            chunk_frames.append(ts)

        chunk_df = pd.concat(chunk_frames, ignore_index=True)
        part_file = partitions_dir / f"part_{chunk_idx:04d}.parquet"
        chunk_df.to_parquet(part_file, index=False, engine="pyarrow")
        logger.info("  Saved partition %d/%d (%d rows) -> %s",
                    chunk_idx + 1, total_chunks, len(chunk_df), part_file.name)

    # Also build a single combined file if manageable (< 2000 patients)
    combined_path = output_dir / "cgm_wearable.parquet"
    if n_patients <= 1000:
        logger.info("Consolidating partitions into %s...", combined_path)
        tables = [pq.read_table(p) for p in sorted(partitions_dir.glob("part_*.parquet"))]
        full_table = pa.concat_tables(tables)
        pq.write_table(full_table, combined_path)
    else:
        logger.info("Cohort size (%d) > 1000: maintaining partitioned dataset in %s",
                    n_patients, partitions_dir)

    return partitions_dir


def main() -> None:
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Generate synthetic EHR + CGM data.")
    parser.add_argument("--cohort_size", "--n_patients", type=int, default=None,
                        help="Number of patients to generate (e.g., 500, 10000).")
    parser.add_argument("--days", type=int, default=None,
                        help="Number of days per patient (default: config value).")
    parser.add_argument("--partitioned", action="store_true",
                        help="Stream outputs into partitioned Parquet chunks.")
    args = parser.parse_args()

    setup_logging(log_file="data_generation.log")
    config = load_config()
    root = get_project_root()

    n_patients = args.cohort_size or config["data"].get("n_patients", 500)
    n_days = args.days or config["data"].get("n_days", 30)

    # Update config copy
    config["data"]["n_patients"] = n_patients
    config["data"]["n_days"] = n_days

    data_dir = root / config["data"]["output_dir"]
    sample_dir = root / config["data"]["sample_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ── Step 1: Generate EHR profiles ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 2: Generating %d EHR profiles (Synthea-calibrated)", n_patients)
    logger.info("=" * 60)
    ehr_df = generate_ehr_profiles(n_patients=n_patients, config=config)
    ehr_path = data_dir / "ehr_profiles.csv"
    ehr_df.to_csv(ehr_path, index=False)
    logger.info("Saved EHR profiles -> %s  (%d rows)", ehr_path, len(ehr_df))

    # ── Step 2: Generate CGM / wearable time-series ────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 2: Generating CGM / wearable time-series (%d days)", n_days)
    logger.info("=" * 60)

    if n_patients > 500 or args.partitioned:
        generate_partitioned_cgm(ehr_df, data_dir, config, chunk_size=100, n_days=n_days)
        cgm_path = data_dir / "cgm_wearable.parquet"
        if not cgm_path.exists():
            cgm_path = data_dir / "cgm_partitions"
    else:
        from src.data_gen.cgm_generator import generate_all_timeseries
        cgm_df = generate_all_timeseries(ehr_df, n_days=n_days, config=config)
        cgm_path = data_dir / "cgm_wearable.parquet"
        cgm_df.to_parquet(cgm_path, index=False, engine="pyarrow")
        logger.info("Saved CGM data  -> %s  (%d rows)", cgm_path, len(cgm_df))

    # ── Sample subset ──────────────────────────────────────────────────────
    sample_n = min(config["data"]["sample_patients"], len(ehr_df))
    sample_pids = ehr_df["patient_id"].iloc[:sample_n].tolist()
    ehr_df[ehr_df["patient_id"].isin(sample_pids)].to_csv(
        sample_dir / "ehr_profiles_sample.csv", index=False,
    )

    sample_frames = []
    for _, prow in ehr_df[ehr_df["patient_id"].isin(sample_pids)].iterrows():
        sample_frames.append(generate_patient_timeseries(prow, n_days=min(14, n_days), config=config))
    sample_cgm = pd.concat(sample_frames, ignore_index=True)
    sample_cgm.to_parquet(
        sample_dir / "cgm_wearable_sample.parquet", index=False, engine="pyarrow",
    )
    logger.info("Saved sample data (%d patients) -> %s", sample_n, sample_dir)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Data generation complete in %.1f seconds.", elapsed)
    logger.info("  EHR profiles : %s", ehr_path)
    logger.info("  CGM/wearable : %s", cgm_path)
    logger.info("=" * 60)

    print("\n-- Generation Summary --")
    print(f"  Patients         : {len(ehr_df):,}")
    print(f"  T2D diagnosed    : {ehr_df['diagnosed_t2d'].sum():,}")
    print(f"  Days per patient : {n_days}")
    print(f"  CGM path         : {cgm_path}")
    print(f"  Time elapsed     : {elapsed:.1f} s")


if __name__ == "__main__":
    main()
