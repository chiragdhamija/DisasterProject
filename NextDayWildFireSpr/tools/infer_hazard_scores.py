#!/usr/bin/env python3
"""Run hazard inference on canonical pickled dataset and export per-sample scores."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "NextDayWildFireSpr"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "next-day-wildfire-spread-ca-hazard"
DEFAULT_CHANNELS_METADATA = DEFAULT_DATASET_DIR / "channels_metadata.json"
DEFAULT_SAMPLE_INDEX = DEFAULT_DATASET_DIR / "sample_index.csv"
DEFAULT_WEIGHTS = PROJECT_DIR / "savedModels" / "model-U_Net-bestF1Score-Rank-0.weights"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data" / "interim" / "hazard_predictions.csv"
DEFAULT_SUMMARY_JSON = PROJECT_DIR / "data" / "interim" / "hazard_predictions_summary.json"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _metrics_from_confusion(tp: int, tn: int, fp: int, fn: int, brier_sum: float, pixel_count: int) -> dict[str, float]:
    precision_1 = _safe_div(tp, tp + fp)
    recall_1 = _safe_div(tp, tp + fn)
    f1_1 = _safe_div(2 * precision_1 * recall_1, precision_1 + recall_1)
    iou_1 = _safe_div(tp, tp + fp + fn)

    precision_0 = _safe_div(tn, tn + fn)
    recall_0 = _safe_div(tn, tn + fp)
    f1_0 = _safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)
    iou_0 = _safe_div(tn, tn + fn + fp)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    macro_f1 = (f1_1 + f1_0) / 2.0
    macro_iou = (iou_1 + iou_0) / 2.0
    macro_precision = (precision_1 + precision_0) / 2.0
    macro_recall = (recall_1 + recall_0) / 2.0

    return {
        "precision": precision_1,
        "recall": recall_1,
        "f1": f1_1,
        "iou": iou_1,
        "accuracy": accuracy,
        "brier_score": _safe_div(brier_sum, pixel_count),
        "pixel_count": int(pixel_count),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "class0_f1": f1_0,
        "class1_f1": f1_1,
    }


def _iter_batches(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


def _load_model(num_input_channels: int, weights_path: Path, device: torch.device):
    from leejunhyun_unet_models import U_Net

    model = U_Net(num_input_channels, 1).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(
    dataset_dir: Path,
    channels_metadata: Path,
    sample_index_csv: Path,
    weights_path: Path,
    output_csv: Path,
    summary_json: Path,
    batch_size: int,
    threshold: float,
    split: str,
    force_cpu: bool,
) -> int:
    if not dataset_dir.exists():
        print(f"[ERROR] dataset_dir not found: {dataset_dir}")
        return 2
    if not channels_metadata.exists():
        print(f"[ERROR] channels_metadata not found: {channels_metadata}")
        return 2
    if not sample_index_csv.exists():
        print(f"[ERROR] sample_index_csv not found: {sample_index_csv}")
        return 2
    if not weights_path.exists():
        print(f"[ERROR] model weights not found: {weights_path}")
        return 2

    metadata = json.loads(channels_metadata.read_text(encoding="utf-8"))
    channel_names = metadata.get("channel_names", [])
    num_channels = len(channel_names)
    if num_channels <= 0:
        print("[ERROR] Invalid channel metadata; no channel_names found")
        return 2

    sample_index = pd.read_csv(sample_index_csv)
    required_cols = {"output_split", "array_index", "sample_id", "sample_date", "sample_lon", "sample_lat"}
    missing_cols = sorted(required_cols - set(sample_index.columns))
    if missing_cols:
        print(f"[ERROR] sample_index missing required columns: {missing_cols}")
        return 2

    splits = ["train", "validation", "test"] if split == "all" else [split]
    output_rows: list[dict[str, object]] = []

    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda:0")
    model = _load_model(num_channels, weights_path=weights_path, device=device)

    # Pixel-level confusion stats (global + per split).
    global_stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "brier_sum": 0.0, "pixel_count": 0}
    split_stats: dict[str, dict[str, float]] = {}

    for out_split in splits:
        data_path = dataset_dir / f"{out_split}.data"
        labels_path = dataset_dir / f"{out_split}.labels"
        if not data_path.exists() or not labels_path.exists():
            print(f"[WARN] Missing split files for {out_split}; skipping")
            continue

        x = _load_pickle(data_path).astype(np.float32, copy=False)
        y = _load_pickle(labels_path).astype(np.float32, copy=False)
        if x.ndim != 4:
            print(f"[ERROR] Invalid data shape for {out_split}: {x.shape}")
            return 2
        if y.ndim != 3:
            print(f"[ERROR] Invalid label shape for {out_split}: {y.shape}")
            return 2
        if x.shape[0] != y.shape[0]:
            print(f"[ERROR] Sample count mismatch in {out_split}: data={x.shape[0]} labels={y.shape[0]}")
            return 2
        if x.shape[1] != num_channels:
            print(
                "[ERROR] Channel count mismatch:"
                f" data has {x.shape[1]}, metadata expects {num_channels}"
            )
            return 2

        split_index = sample_index[sample_index["output_split"] == out_split].copy()
        split_index["array_index"] = pd.to_numeric(split_index["array_index"], errors="coerce")
        split_index = split_index.dropna(subset=["array_index"])
        split_index["array_index"] = split_index["array_index"].astype(int)
        split_index_map = split_index.set_index("array_index").to_dict(orient="index")

        n = x.shape[0]
        print(f"[INFO] Running split={out_split}, samples={n}, device={device.type}")
        split_stats[out_split] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "brier_sum": 0.0, "pixel_count": 0}

        with torch.no_grad():
            for i0, i1 in _iter_batches(n, batch_size):
                xb = torch.from_numpy(x[i0:i1]).to(device)
                yb = torch.from_numpy(y[i0:i1]).to(device)

                logits = model(xb).squeeze(1)
                probs = torch.sigmoid(logits)
                pred = (probs >= threshold).float()

                # Sample-level exported hazard summaries.
                probs_flat = probs.flatten(1)
                pred_flat = pred.flatten(1)
                y_flat = yb.flatten(1)
                prob_mean = probs_flat.mean(dim=1).cpu().numpy()
                prob_p95 = torch.quantile(probs_flat, q=0.95, dim=1).cpu().numpy()
                prob_max = probs_flat.max(dim=1).values.cpu().numpy()
                pred_fire_frac = pred_flat.mean(dim=1).cpu().numpy()
                gt_fire_frac = y_flat.mean(dim=1).cpu().numpy()

                for j, arr_idx in enumerate(range(i0, i1)):
                    meta = split_index_map.get(arr_idx, {})
                    output_rows.append(
                        {
                            "output_split": out_split,
                            "array_index": int(arr_idx),
                            "sample_id": meta.get("sample_id"),
                            "sample_date": meta.get("sample_date"),
                            "sample_lon": meta.get("sample_lon"),
                            "sample_lat": meta.get("sample_lat"),
                            "hazard_prob_mean": float(prob_mean[j]),
                            "hazard_prob_p95": float(prob_p95[j]),
                            "hazard_prob_max": float(prob_max[j]),
                            "hazard_pred_fire_frac": float(pred_fire_frac[j]),
                            "gt_fire_frac": float(gt_fire_frac[j]),
                        }
                    )

                # Global pixel metrics.
                y_bin = (yb > 0.5).float()
                tp = int(((pred == 1) & (y_bin == 1)).sum().item())
                tn = int(((pred == 0) & (y_bin == 0)).sum().item())
                fp = int(((pred == 1) & (y_bin == 0)).sum().item())
                fn = int(((pred == 0) & (y_bin == 1)).sum().item())
                brier = float(((probs - y_bin) ** 2).sum().item())
                pcount = int(y_bin.numel())

                for stats in (global_stats, split_stats[out_split]):
                    stats["tp"] += tp
                    stats["tn"] += tn
                    stats["fp"] += fp
                    stats["fn"] += fn
                    stats["brier_sum"] += brier
                    stats["pixel_count"] += pcount

    if not output_rows:
        print("[ERROR] No predictions were produced.")
        return 2

    out_df = pd.DataFrame(output_rows)
    out_df = out_df.sort_values(["output_split", "array_index"], kind="mergesort").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    global_metrics = _metrics_from_confusion(
        tp=int(global_stats["tp"]),
        tn=int(global_stats["tn"]),
        fp=int(global_stats["fp"]),
        fn=int(global_stats["fn"]),
        brier_sum=float(global_stats["brier_sum"]),
        pixel_count=int(global_stats["pixel_count"]),
    )
    split_metrics = {}
    for out_split, stats in split_stats.items():
        split_metrics[out_split] = _metrics_from_confusion(
            tp=int(stats["tp"]),
            tn=int(stats["tn"]),
            fp=int(stats["fp"]),
            fn=int(stats["fn"]),
            brier_sum=float(stats["brier_sum"]),
            pixel_count=int(stats["pixel_count"]),
        )

    summary = {
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "channels_metadata": str(channels_metadata),
            "sample_index_csv": str(sample_index_csv),
            "weights_path": str(weights_path),
            "split": split,
            "threshold": threshold,
            "batch_size": batch_size,
            "device": device.type,
        },
        "outputs": {
            "predictions_csv": str(output_csv),
            "rows": int(len(out_df)),
            "split_counts": out_df["output_split"].value_counts(dropna=False).to_dict(),
        },
        "hazard_distribution": {
            "mean_of_mean_prob": float(out_df["hazard_prob_mean"].mean()),
            "p95_of_mean_prob": float(out_df["hazard_prob_mean"].quantile(0.95)),
            "max_prob_observed": float(out_df["hazard_prob_max"].max()),
            "mean_pred_fire_frac": float(out_df["hazard_pred_fire_frac"].mean()),
        },
        "pixel_metrics": {
            "threshold": threshold,
            "precision": global_metrics["precision"],
            "recall": global_metrics["recall"],
            "f1": global_metrics["f1"],
            "iou": global_metrics["iou"],
            "accuracy": global_metrics["accuracy"],
            "brier_score": global_metrics["brier_score"],
            "pixel_count": global_metrics["pixel_count"],
        },
        "pixel_metrics_macro_like_training": {
            "macro_precision": global_metrics["macro_precision"],
            "macro_recall": global_metrics["macro_recall"],
            "macro_f1": global_metrics["macro_f1"],
            "macro_iou": global_metrics["macro_iou"],
            "class0_f1": global_metrics["class0_f1"],
            "class1_f1": global_metrics["class1_f1"],
        },
        "per_split_metrics": {
            split_name: {
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1"],
                "iou": vals["iou"],
                "accuracy": vals["accuracy"],
                "macro_f1": vals["macro_f1"],
                "macro_iou": vals["macro_iou"],
                "class0_f1": vals["class0_f1"],
                "class1_f1": vals["class1_f1"],
                "pixel_count": vals["pixel_count"],
            }
            for split_name, vals in split_metrics.items()
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote hazard predictions: {output_csv}")
    print(f"[DONE] Wrote hazard summary: {summary_json}")
    print(
        "[INFO] Pixel metrics:"
        f" PosF1={summary['pixel_metrics']['f1']:.4f},"
        f" MacroF1={summary['pixel_metrics_macro_like_training']['macro_f1']:.4f},"
        f" MacroIoU={summary['pixel_metrics_macro_like_training']['macro_iou']:.4f},"
        f" Acc={summary['pixel_metrics']['accuracy']:.4f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--channels_metadata", type=Path, default=DEFAULT_CHANNELS_METADATA)
    parser.add_argument("--sample_index_csv", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--split",
        choices=["all", "train", "validation", "test"],
        default="all",
        help="Which output split(s) to score",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_inference(
        dataset_dir=args.dataset_dir,
        channels_metadata=args.channels_metadata,
        sample_index_csv=args.sample_index_csv,
        weights_path=args.weights,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        batch_size=args.batch_size,
        threshold=args.threshold,
        split=args.split,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
