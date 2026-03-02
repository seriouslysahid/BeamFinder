"""
YOLO26 Model Comparison Study

Trains all five YOLO26 detection variants on the DeepSense Scenario 23 drone
dataset (100 epochs each) and compares accuracy, training time, and inference
speed.  Results are saved to runs/study/ as JSON + charts.

Supports crash recovery: if the kernel/process restarts, re-running the script
will skip models that already finished.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# ── Models to compare ──────────────────────────────────────
MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
LABEL_MAP = {"yolo26n": "Nano", "yolo26s": "Small", "yolo26m": "Medium", "yolo26l": "Large", "yolo26x": "XLarge"}

# ── Shared training args (tuned for A100 40/80 GB) ────────
TRAIN_ARGS = dict(
    data="data.yaml", epochs=100, imgsz=960, batch=0.90,
    patience=20, cache="ram", workers=8, cos_lr=True,
    deterministic=False, rect=True, save_period=10,
    degrees=15.0, flipud=0.5, scale=0.9, translate=0.2,
)

# ── Output paths ───────────────────────────────────────────
RESULTS_FILE = Path("runs/study/results_summary.json")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  Training loop with crash recovery
# ═══════════════════════════════════════════════════════════
def train_all():
    # A100: enable TF32 for ~3x faster matmuls with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resume from previous results if process restarted
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        done = {r["model"] for r in results}
        print(f"Resuming — {len(done)} models already done: {done}")
    else:
        results = []
        done = set()

    for i, pt_file in enumerate(MODELS):
        name = Path(pt_file).stem
        run_dir = Path(f"runs/study/{name}")

        if name in done:
            print(f"\n[{i+1}/{len(MODELS)}] {name} — SKIPPED (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(MODELS)}] Training {name}")
        print(f"{'='*60}\n")

        model = YOLO(pt_file)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

        t0 = time.time()
        model.train(**TRAIN_ARGS, project="runs/study", name=name, exist_ok=True)
        train_time = time.time() - t0

        # Load best weights for evaluation
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            model = YOLO(str(best_pt))

        val = model.val(data="data.yaml", imgsz=960, half=True)
        test = model.val(data="data.yaml", split="test", imgsz=960, half=True)

        results.append({
            "model": name,
            "params_M": round(params, 1),
            "train_time_min": round(train_time / 60, 1),
            "val_mAP50": round(val.box.map50, 4),
            "val_mAP50_95": round(val.box.map, 4),
            "test_mAP50": round(test.box.map50, 4),
            "test_mAP50_95": round(test.box.map, 4),
            "inference_ms": round(val.speed["inference"], 2),
        })
        done.add(name)

        # Save after each model (crash recovery)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"\n✓ {name} done — saved to {RESULTS_FILE}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All {len(MODELS)} models completed!")
    print(f"{'='*60}")
    return results


# ═══════════════════════════════════════════════════════════
#  Comparison bar charts
# ═══════════════════════════════════════════════════════════
def plot_comparison(results):
    labels = [LABEL_MAP.get(r["model"], r["model"]) for r in results]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#DDA0DD"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLO26 Model Comparison — Drone Detection (100 Epochs, imgsz=960)",
                 fontsize=15, fontweight="bold")

    x = range(len(results))
    w = 0.35

    # ---- mAP@50 ----
    ax = axes[0, 0]
    v50 = [r["val_mAP50"] for r in results]
    t50 = [r["test_mAP50"] for r in results]
    b1 = ax.bar([i - w/2 for i in x], v50, w, label="Val", color="#4CAF50", edgecolor="white")
    b2 = ax.bar([i + w/2 for i in x], t50, w, label="Test", color="#2196F3", edgecolor="white")
    ax.set_title("mAP@50", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.legend(); ax.set_ylim(0, 1)
    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)

    # ---- mAP@50-95 ----
    ax = axes[0, 1]
    v95 = [r["val_mAP50_95"] for r in results]
    t95 = [r["test_mAP50_95"] for r in results]
    b1 = ax.bar([i - w/2 for i in x], v95, w, label="Val", color="#4CAF50", edgecolor="white")
    b2 = ax.bar([i + w/2 for i in x], t95, w, label="Test", color="#2196F3", edgecolor="white")
    ax.set_title("mAP@50-95", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.legend(); ax.set_ylim(0, 1)
    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)

    # ---- Training Time ----
    ax = axes[1, 0]
    times = [r["train_time_min"] for r in results]
    bars = ax.bar(x, times, color=colors[:len(results)], edgecolor="white")
    ax.set_title("Training Time", fontweight="bold")
    ax.set_ylabel("Minutes")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=2)

    # ---- Inference Speed ----
    ax = axes[1, 1]
    speeds = [r["inference_ms"] for r in results]
    bars = ax.bar(x, speeds, color=colors[:len(results)], edgecolor="white")
    ax.set_title("Inference Speed", fontweight="bold")
    ax.set_ylabel("ms / image")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)

    plt.tight_layout()
    out = "runs/study/comparison_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")


# ═══════════════════════════════════════════════════════════
#  Efficiency scatter plots
# ═══════════════════════════════════════════════════════════
def plot_efficiency(results):
    labels = [LABEL_MAP.get(r["model"], r["model"]) for r in results]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#DDA0DD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("YOLO26 Efficiency Analysis", fontsize=14, fontweight="bold")

    for i, r in enumerate(results):
        ax1.scatter(r["params_M"], r["test_mAP50_95"], s=180, c=colors[i],
                    edgecolors="black", zorder=3)
        ax1.annotate(labels[i], (r["params_M"], r["test_mAP50_95"]),
                     textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
    ax1.set_xlabel("Parameters (M)", fontsize=12)
    ax1.set_ylabel("Test mAP@50-95", fontsize=12)
    ax1.set_title("Accuracy vs Model Size")
    ax1.grid(True, alpha=0.3)

    for i, r in enumerate(results):
        ax2.scatter(r["inference_ms"], r["test_mAP50_95"], s=180, c=colors[i],
                    edgecolors="black", zorder=3)
        ax2.annotate(labels[i], (r["inference_ms"], r["test_mAP50_95"]),
                     textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
    ax2.set_xlabel("Inference Time (ms/image)", fontsize=12)
    ax2.set_ylabel("Test mAP@50-95", fontsize=12)
    ax2.set_title("Accuracy vs Speed")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "runs/study/efficiency_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = train_all()
    plot_comparison(results)
    plot_efficiency(results)
