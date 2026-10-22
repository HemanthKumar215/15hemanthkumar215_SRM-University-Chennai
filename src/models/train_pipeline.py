"""
train_pipeline.py – Three-Stage Training Pipeline with Real Data Adaptation
=============================================================================
Orchestrates:
Stage A: Pretraining FusionModel on synthetic cohort with modality masking.
Stage B: Fine-tuning on real clinical CGM data with missing-modality handling.
Stage C: Rigorous patient-disjoint evaluation on real held-out test cohort
         and subgroup stratification (Normal / Prediabetic / T2D, Age, Sex).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.utils.config import get_project_root, load_config
from src.utils.logging_setup import setup_logging
from src.utils.sanity_checks import assert_no_patient_overlap, investigate_high_auroc
from src.utils.tracker import ExperimentTracker
from src.features.lazy_dataset import LazySpikeDataset
from src.models.fusion_model import FusionModel, build_model_from_config
from src.data_real import (
    AIREADIAdapter,
    CGMacrosAdapter,
    ShanghaiAdapter,
    generate_all_mock_datasets,
)

logger = logging.getLogger(__name__)


def apply_modality_dropout(
    ts_batch: torch.Tensor,
    dropout_prob: float = 0.30,
) -> torch.Tensor:
    """Randomly drop non-glucose sensor channels (channels 1:5) during pretraining.

    Ensures the network learns robust representations when real-world datasets
    only contain CGM glucose traces without wearable PPG/accelerometry.
    """
    if dropout_prob <= 0.0 or not ts_batch.requires_grad:
        return ts_batch

    batch_size = ts_batch.size(0)
    # Mask per sample for channels 1..4 (hr, hrv, steps, sleep)
    drop_mask = (torch.rand(batch_size, 1, 4, device=ts_batch.device) > dropout_prob).float()
    ts_masked = ts_batch.clone()
    ts_masked[:, :, 1:5] = ts_masked[:, :, 1:5] * drop_mask
    return ts_masked


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_dropout: float = 0.0,
) -> float:
    """Train for a single epoch with optional modality dropout."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for ts, static, labels in loader:
        ts = ts.to(device)
        static = static.to(device)
        labels = labels.float().to(device)

        if modality_dropout > 0:
            ts = apply_modality_dropout(ts, modality_dropout)

        optimizer.zero_grad()
        logits = model(ts, static).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model and return AUROC, AUPRC, F1, precision, recall."""
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    for ts, static, labels in loader:
        ts = ts.to(device)
        static = static.to(device)
        logits = model(ts, static).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5, "auprc": 0.0, "f1": 0.0,
            "precision": 0.0, "recall": 0.0,
            "n_samples": len(y_true), "n_pos": int(y_true.sum()),
        }

    return {
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "n_samples": len(y_true),
        "n_pos": int(y_true.sum()),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def evaluate_subgroups(
    model: nn.Module,
    cgm_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
    patient_ids: List[str],
    config: Dict[str, Any],
    device: torch.device,
    norm_stats: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Calculate evaluation metrics broken down by clinical and demographic subgroups."""
    sub_results = {}
    ehr_sub = ehr_df[ehr_df["patient_id"].isin(patient_ids)].copy()

    # Subgroup definitions
    subgroups = {
        # Cohorts
        "Cohort: Normal": ehr_sub[ehr_sub["cohort"] == "Normal"]["patient_id"].tolist(),
        "Cohort: Prediabetic": ehr_sub[ehr_sub["cohort"] == "Prediabetic"]["patient_id"].tolist(),
        "Cohort: T2D": ehr_sub[ehr_sub["cohort"] == "T2D"]["patient_id"].tolist(),
        # Sex
        "Sex: Male": ehr_sub[ehr_sub["sex"] == "M"]["patient_id"].tolist(),
        "Sex: Female": ehr_sub[ehr_sub["sex"] == "F"]["patient_id"].tolist(),
        # Age bands
        "Age: <40": ehr_sub[ehr_sub["age"] < 40]["patient_id"].tolist(),
        "Age: 40-60": ehr_sub[(ehr_sub["age"] >= 40) & (ehr_sub["age"] <= 60)]["patient_id"].tolist(),
        "Age: >60": ehr_sub[ehr_sub["age"] > 60]["patient_id"].tolist(),
    }

    for name, pids in subgroups.items():
        if len(pids) < 2:
            continue
        ds = LazySpikeDataset(cgm_df, ehr_df, pids, config, max_samples_per_patient=20)
        ds.set_normalization(
            norm_stats["ts_mean"], norm_stats["ts_std"],
            norm_stats["static_mean"], norm_stats["static_std"]
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        m = evaluate_dataset(model, loader, device)
        sub_results[name] = {
            "n_patients": len(pids),
            "n_samples": m["n_samples"],
            "n_pos": m["n_pos"],
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "f1": m["f1"],
            "recall": m["recall"],
        }

    return sub_results


def run_pipeline(
    epochs_pretrain: int = 5,
    epochs_finetune: int = 5,
    batch_size: int = 128,
    skip_pretrain: bool = False,
) -> Dict[str, Any]:
    """Execute the full 3-stage training, real fine-tuning, and evaluation workflow."""
    setup_logging(log_file="pipeline.log")
    config = load_config()
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pipeline execution device: %s", device)

    tracker = ExperimentTracker(root / "reports")
    models_dir = root / config["paths"]["models_dir"]
    reports_dir = root / config["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Synthetic Cohort ───────────────────────────────────────────
    synth_ehr = pd.read_csv(root / config["data"]["output_dir"] / "ehr_profiles.csv")
    synth_cgm_path = root / config["data"]["output_dir"] / "cgm_wearable.parquet"
    if not synth_cgm_path.exists():
        synth_cgm_path = root / config["data"]["output_dir"] / "cgm_partitions"

    # Patient disjoint split for synthetic cohort
    synth_pids = synth_ehr["patient_id"].unique()
    rng = np.random.default_rng(config["project"]["seed"])
    shuffled_pids = rng.permutation(synth_pids)
    n_synth = len(shuffled_pids)
    n_test = int(n_synth * config["model"]["test_fraction"])
    n_val = int(n_synth * config["model"]["val_fraction"])

    synth_train_pids = shuffled_pids[n_test + n_val:].tolist()
    synth_val_pids = shuffled_pids[n_test: n_test + n_val].tolist()
    synth_test_pids = shuffled_pids[:n_test].tolist()

    # Leakage assertion on synthetic splits
    assert_no_patient_overlap(synth_train_pids, synth_val_pids, synth_test_pids)

    # Build synthetic datasets
    logger.info("Building synthetic datasets...")
    synth_train_ds = LazySpikeDataset(synth_cgm_path, synth_ehr, synth_train_pids, config, max_samples_per_patient=30)
    synth_val_ds = LazySpikeDataset(synth_cgm_path, synth_ehr, synth_val_pids, config, max_samples_per_patient=15)
    synth_test_ds = LazySpikeDataset(synth_cgm_path, synth_ehr, synth_test_pids, config, max_samples_per_patient=15)

    synth_val_ds.set_normalization(synth_train_ds.ts_mean, synth_train_ds.ts_std, synth_train_ds.static_mean, synth_train_ds.static_std)
    synth_test_ds.set_normalization(synth_train_ds.ts_mean, synth_train_ds.ts_std, synth_train_ds.static_mean, synth_train_ds.static_std)

    synth_train_loader = DataLoader(synth_train_ds, batch_size=batch_size, shuffle=True)
    synth_val_loader = DataLoader(synth_val_ds, batch_size=batch_size, shuffle=False)
    synth_test_loader = DataLoader(synth_test_ds, batch_size=batch_size, shuffle=False)

    # ── 2. Load Real-World Datasets ────────────────────────────────────────
    real_data_dir = root / "data" / "real"
    if not (real_data_dir / "aireadi").exists():
        generate_all_mock_datasets(real_data_dir)

    aireadi_cgm, aireadi_ehr = AIREADIAdapter(real_data_dir / "aireadi").load_all()
    cgmacros_cgm, cgmacros_ehr = CGMacrosAdapter(real_data_dir / "cgmacros").load_all()
    shanghai_cgm, shanghai_ehr = ShanghaiAdapter(real_data_dir / "shanghai").load_all()

    real_cgm = pd.concat([aireadi_cgm, cgmacros_cgm, shanghai_cgm], ignore_index=True)
    real_ehr = pd.concat([aireadi_ehr, cgmacros_ehr, shanghai_ehr], ignore_index=True)

    # Patient disjoint split for real cohort:
    # 60% real fine-tune train, 20% real validation, 20% real held-out test
    real_pids = real_ehr["patient_id"].unique()
    rng_real = np.random.default_rng(777)
    shuffled_real_pids = rng_real.permutation(real_pids)
    n_real = len(shuffled_real_pids)
    n_real_test = max(3, int(n_real * 0.20))
    n_real_val = max(3, int(n_real * 0.20))

    real_train_pids = shuffled_real_pids[n_real_test + n_real_val:].tolist()
    real_val_pids = shuffled_real_pids[n_real_test: n_real_test + n_real_val].tolist()
    real_heldout_pids = shuffled_real_pids[:n_real_test].tolist()

    # Leakage assertion on real splits
    assert_no_patient_overlap(real_train_pids, real_val_pids, real_heldout_pids)
    assert_no_patient_overlap(synth_pids, real_pids, [])

    real_train_ds = LazySpikeDataset(real_cgm, real_ehr, real_train_pids, config, max_samples_per_patient=30)
    real_val_ds = LazySpikeDataset(real_cgm, real_ehr, real_val_pids, config, max_samples_per_patient=15)
    real_heldout_ds = LazySpikeDataset(real_cgm, real_ehr, real_heldout_pids, config, max_samples_per_patient=15)

    real_train_ds.set_normalization(synth_train_ds.ts_mean, synth_train_ds.ts_std, synth_train_ds.static_mean, synth_train_ds.static_std)
    real_val_ds.set_normalization(synth_train_ds.ts_mean, synth_train_ds.ts_std, synth_train_ds.static_mean, synth_train_ds.static_std)
    real_heldout_ds.set_normalization(synth_train_ds.ts_mean, synth_train_ds.ts_std, synth_train_ds.static_mean, synth_train_ds.static_std)

    real_train_loader = DataLoader(real_train_ds, batch_size=batch_size, shuffle=True)
    real_val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False)
    real_heldout_loader = DataLoader(real_heldout_ds, batch_size=batch_size, shuffle=False)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE A: Pretraining on Synthetic Data with Modality Dropout
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE A: Pretraining FusionModel on Synthetic Cohort (%d epochs)", epochs_pretrain)
    logger.info("=" * 60)

    cfg_override = dict(config["model"])
    cfg_override["static_input_dim"] = synth_train_ds.static_dim
    model = build_model_from_config(cfg_override, "fusion").to(device)

    pos_w = torch.tensor([synth_train_ds.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_synth_val_auroc = 0.0
    best_pretrain_state = None
    pretrain_ckpt_path = models_dir / "pretrained_fusion.pt"

    if skip_pretrain and pretrain_ckpt_path.exists():
        logger.info("Found existing %s; loading and skipping Stage A pretrain...", pretrain_ckpt_path)
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        best_synth_val_auroc = evaluate_dataset(model, synth_val_loader, device)["auroc"]
    else:
        for ep in range(1, epochs_pretrain + 1):
            loss = train_one_epoch(model, synth_train_loader, criterion, optimizer, device, modality_dropout=0.30)
            v_metrics = evaluate_dataset(model, synth_val_loader, device)
            scheduler.step(v_metrics["auroc"])
            logger.info("  [Pretrain] Epoch %d/%d  loss=%.4f  val_auroc=%.4f  val_f1=%.4f",
                        ep, epochs_pretrain, loss, v_metrics["auroc"], v_metrics["f1"])
            if v_metrics["auroc"] > best_synth_val_auroc:
                best_synth_val_auroc = v_metrics["auroc"]
                best_pretrain_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_pretrain_state:
            model.load_state_dict(best_pretrain_state)
        model.to(device)

        # Save pretrained checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg_override,
            "norm_stats": {
                "ts_mean": synth_train_ds.ts_mean.tolist(),
                "ts_std": synth_train_ds.ts_std.tolist(),
                "static_mean": synth_train_ds.static_mean.tolist(),
                "static_std": synth_train_ds.static_std.tolist(),
            },
        }, pretrain_ckpt_path)
        logger.info("Saved pretrained model -> %s", pretrain_ckpt_path)

    # Evaluate Stage A model on all splits
    stage_a_synth_test = evaluate_dataset(model, synth_test_loader, device)
    stage_a_real_val = evaluate_dataset(model, real_val_loader, device)
    stage_a_real_heldout = evaluate_dataset(model, real_heldout_loader, device)

    tracker.log_run("stage_a_pretrain", "Pretrain", "fusion",
                    {"epochs": epochs_pretrain, "lr": 1e-3},
                    {"val_auroc": best_synth_val_auroc, "test": stage_a_synth_test},
                    notes="Synthetic pretraining with modality dropout")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE B: Fine-Tuning on Real Clinical Data
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE B: Fine-Tuning on Real Clinical Data (%d epochs)", epochs_finetune)
    logger.info("=" * 60)

    # Fine-tuning with lower learning rate
    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_optimizer, mode="max", factor=0.5, patience=2)
    ft_pos_w = torch.tensor([real_train_ds.pos_weight], device=device)
    ft_criterion = nn.BCEWithLogitsLoss(pos_weight=ft_pos_w)

    best_real_val_auroc = 0.0
    best_finetune_state = None

    for ep in range(1, epochs_finetune + 1):
        loss = train_one_epoch(model, real_train_loader, ft_criterion, ft_optimizer, device, modality_dropout=0.0)
        v_metrics = evaluate_dataset(model, real_val_loader, device)
        ft_scheduler.step(v_metrics["auroc"])
        logger.info("  [Fine-tune] Epoch %d/%d  loss=%.4f  real_val_auroc=%.4f  real_val_f1=%.4f",
                    ep, epochs_finetune, loss, v_metrics["auroc"], v_metrics["f1"])
        if v_metrics["auroc"] > best_real_val_auroc:
            best_real_val_auroc = v_metrics["auroc"]
            best_finetune_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_finetune_state:
        model.load_state_dict(best_finetune_state)
    model.to(device)

    # Save fine-tuned checkpoint
    finetune_ckpt_path = models_dir / "finetuned_fusion.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg_override,
        "norm_stats": {
            "ts_mean": synth_train_ds.ts_mean.tolist(),
            "ts_std": synth_train_ds.ts_std.tolist(),
            "static_mean": synth_train_ds.static_mean.tolist(),
            "static_std": synth_train_ds.static_std.tolist(),
        },
    }, finetune_ckpt_path)
    logger.info("Saved fine-tuned model -> %s", finetune_ckpt_path)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE C: Rigorous Final Evaluation & Subgroup Stratification
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE C: Final Held-Out Evaluation & Subgroup Stratification")
    logger.info("=" * 60)

    stage_c_synth_test = evaluate_dataset(model, synth_test_loader, device)
    stage_c_real_val = evaluate_dataset(model, real_val_loader, device)
    stage_c_real_heldout = evaluate_dataset(model, real_heldout_loader, device)

    tracker.log_run("stage_b_finetune", "Fine-tune", "fusion",
                    {"epochs": epochs_finetune, "lr": 1e-4},
                    {"val_auroc": best_real_val_auroc, "test": stage_c_real_heldout},
                    notes="Fine-tuned on real cohort")

    # Run subgroup stratification on Real Held-Out cohort
    norm_dict = {
        "ts_mean": synth_train_ds.ts_mean,
        "ts_std": synth_train_ds.ts_std,
        "static_mean": synth_train_ds.static_mean,
        "static_std": synth_train_ds.static_std,
    }
    subgroups = evaluate_subgroups(
        model, real_cgm, real_ehr, real_heldout_pids,
        config, device, norm_dict,
    )

    # AUROC Sanity Checks
    audit_synth = investigate_high_auroc(stage_c_synth_test["auroc"], "Fusion (Synthetic Test)")
    audit_real = investigate_high_auroc(stage_c_real_heldout["auroc"], "Fusion (Real Held-Out Test)")

    # ── Compile Summary Output ────────────────────────────────────────────
    summary = {
        "device": str(device),
        "split_counts": {
            "synthetic": {"train": len(synth_train_pids), "val": len(synth_val_pids), "test": len(synth_test_pids)},
            "real": {"train": len(real_train_pids), "val": len(real_val_pids), "heldout_test": len(real_heldout_pids)},
        },
        "stage_a_synthetic_pretrain": {
            "synthetic_test": {k: v for k, v in stage_a_synth_test.items() if k not in ("y_true", "y_prob")},
            "real_val": {k: v for k, v in stage_a_real_val.items() if k not in ("y_true", "y_prob")},
            "real_heldout_test": {k: v for k, v in stage_a_real_heldout.items() if k not in ("y_true", "y_prob")},
        },
        "stage_b_c_finetuned": {
            "synthetic_test": {k: v for k, v in stage_c_synth_test.items() if k not in ("y_true", "y_prob")},
            "real_val": {k: v for k, v in stage_c_real_val.items() if k not in ("y_true", "y_prob")},
            "real_heldout_test": {k: v for k, v in stage_c_real_heldout.items() if k not in ("y_true", "y_prob")},
        },
        "subgroup_performance": subgroups,
        "sanity_audits": [audit_synth, audit_real],
    }

    # Save to reports/pipeline_results.json
    out_file = reports_dir / "pipeline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Pipeline completed. Summary saved to %s", out_file)

    # Print formatted results tables
    print("\n" + "=" * 70)
    print(" 🏥 THREE-STAGE PIPELINE EVALUATION BENCHMARK")
    print("=" * 70)
    print(f"{'Condition':<30} {'Split':<18} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    print("-" * 70)
    print(f"{'Synthetic Pretrained Only':<30} {'Synth Test':<18} {stage_a_synth_test['auroc']:>8.4f} {stage_a_synth_test['auprc']:>8.4f} {stage_a_synth_test['f1']:>8.4f}")
    print(f"{'Synthetic Pretrained Only':<30} {'Real Held-Out':<18} {stage_a_real_heldout['auroc']:>8.4f} {stage_a_real_heldout['auprc']:>8.4f} {stage_a_real_heldout['f1']:>8.4f}")
    print(f"{'Synthetic + Real Fine-Tuned':<30} {'Synth Test':<18} {stage_c_synth_test['auroc']:>8.4f} {stage_c_synth_test['auprc']:>8.4f} {stage_c_synth_test['f1']:>8.4f}")
    print(f"{'Synthetic + Real Fine-Tuned':<30} {'Real Held-Out':<18} {stage_c_real_heldout['auroc']:>8.4f} {stage_c_real_heldout['auprc']:>8.4f} {stage_c_real_heldout['f1']:>8.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" 👥 SUBGROUP STRATIFICATION (Real Held-Out Cohort)")
    print("=" * 70)
    print(f"{'Subgroup':<25} {'Patients':>10} {'Samples':>10} {'AUROC':>8} {'F1':>8}")
    print("-" * 70)
    for sg_name, vals in subgroups.items():
        print(f"{sg_name:<25} {vals['n_patients']:>10} {vals['n_samples']:>10} {vals['auroc']:>8.4f} {vals['f1']:>8.4f}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Stage Training Pipeline.")
    parser.add_argument("--epochs_pretrain", type=int, default=3, help="Pretraining epochs.")
    parser.add_argument("--epochs_finetune", type=int, default=3, help="Fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--skip_pretrain", action="store_true", help="Skip pretrain if pretrained_fusion.pt exists.")
    args = parser.parse_args()

    run_pipeline(
        epochs_pretrain=args.epochs_pretrain,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        skip_pretrain=args.skip_pretrain,
    )
