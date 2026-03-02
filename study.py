"""
YOLO26 Model Comparison Study

Trains all five YOLO26 detection variants on the DeepSense Scenario 23 drone
dataset (100 epochs each) and compares:
  - Training time
  - Val / Test accuracy (mAP@50, mAP@50-95)
  - Inference speed breakdown (preprocess, inference, postprocess)
  - Peak GPU memory during training

Results are saved to runs/study/ as JSON + charts.

Supports crash recovery: if the process restarts, re-running the script
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

# ── Paths (absolute, avoids Ultralytics path nesting) ─────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR / "runs" / "study"
RESULTS_FILE = STUDY_DIR / "results_summary.json"
STUDY_DIR.mkdir(parents=True, exist_ok=True)

# ── Models to compare ──────────────────────────────────────
MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
LABEL_MAP = {
    "yolo26n": "Nano", "yolo26s": "Small", "yolo26m": "Medium",
    "yolo26l": "Large", "yolo26x": "XLarge",
}

# ── Shared training args (tuned for A100 40 GB) ───────────
TRAIN_ARGS = dict(
    data="data.yaml", epochs=100, imgsz=960, batch=0.90,
    patience=20, cache="ram", workers=8, cos_lr=True,
    deterministic=False, rect=True, save_period=10,
    compile=True,       # torch.compile — 10-30% faster on A100
    degrees=15.0, flipud=0.5, scale=0.9, translate=0.2,
)


# ═══════════════════════════════════════════════════════════
#  Pre-download all model weights
# ═══════════════════════════════════════════════════════════
def download_all_weights():
    """Download all pretrained weights up front so training isn't stalled."""
    print("Pre-downloading model weights...")
    for pt_file in MODELS:
        if not Path(pt_file).exists():
            YOLO(pt_file)  # triggers auto-download
            print(f"  ✓ {pt_file}")
        else:
            print(f"  ✓ {pt_file} (cached)")


# ═══════════════════════════════════════════════════════════
#  Training loop with crash recovery
# ═══════════════════════════════════════════════════════════
def train_all():
    # A100: maximize GPU throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # auto-tune convolutions for fixed imgsz

    download_all_weights()

    # Resume from previous results if process restarted
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        done = {r["model"] for r in results}
        print(f"\nResuming — {len(done)} models already done: {done}")
    else:
        results = []
        done = set()

    for i, pt_file in enumerate(MODELS):
        name = Path(pt_file).stem
        run_dir = STUDY_DIR / name

        if name in done:
            print(f"\n[{i+1}/{len(MODELS)}] {name} — SKIPPED (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(MODELS)}] Training {name}")
        print(f"{'='*60}\n")

        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()

        model = YOLO(pt_file)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

        t0 = time.time()
        model.train(**TRAIN_ARGS, project=str(STUDY_DIR), name=name, exist_ok=True)
        train_time = time.time() - t0

        train_peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        # Load best weights for evaluation
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            model = YOLO(str(best_pt))

        # Validation metrics
        val = model.val(data="data.yaml", imgsz=960, half=True)

        # Test metrics
        test = model.val(data="data.yaml", split="test", imgsz=960, half=True)

        # Speed breakdown (preprocess / inference / postprocess in ms)
        speed = val.speed

        results.append({
            "model": name,
            "params_M": round(params, 1),
            "train_time_min": round(train_time / 60, 1),
            "train_peak_mem_GB": round(train_peak_mem_gb, 1),
            "val_mAP50": round(val.box.map50, 4),
            "val_mAP50_95": round(val.box.map, 4),
            "test_mAP50": round(test.box.map50, 4),
            "test_mAP50_95": round(test.box.map, 4),
            "preprocess_ms": round(speed.get("preprocess", 0), 2),
            "inference_ms": round(speed.get("inference", 0), 2),
            "postprocess_ms": round(speed.get("postprocess", 0), 2),
        })
        done.add(name)

        # Save after each model (crash recovery)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

        # Print summary for this model
        r = results[-1]
        print(f"\n{'─'*50}")
        print(f"✓ {name} complete")
        print(f"  Params:       {r['params_M']}M")
        print(f"  Train time:   {r['train_time_min']} min")
        print(f"  Peak VRAM:    {r['train_peak_mem_GB']} GB")
        print(f"  Val  mAP50:   {r['val_mAP50']}  mAP50-95: {r['val_mAP50_95']}")
        print(f"  Test mAP50:   {r['test_mAP50']}  mAP50-95: {r['test_mAP50_95']}")
        print(f"  Speed:        {r['preprocess_ms']}ms pre + {r['inference_ms']}ms inf + {r['postprocess_ms']}ms post")
        print(f"{'─'*50}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All {len(MODELS)} models completed!")
    print(f"{'='*60}")

    print_summary_table(results)
    return results


# ═══════════════════════════════════════════════════════════
#  Console summary table
# ═══════════════════════════════════════════════════════════
def print_summary_table(results):
    header = (
        f"\n{'Model':<10} {'Params':>7} {'Train':>7} {'VRAM':>6} "
        f"{'Val50':>7} {'Val95':>7} {'Test50':>7} {'Test95':>7} "
        f"{'Pre':>6} {'Inf':>6} {'Post':>6}"
    )
    units = (
        f"{'':10} {'(M)':>7} {'(min)':>7} {'(GB)':>6} "
        f"{'':>7} {'':>7} {'':>7} {'':>7} "
        f"{'(ms)':>6} {'(ms)':>6} {'(ms)':>6}"
    )
    print(header)
    print(units)
    print("─" * len(header.strip()))
    for r in results:
        print(
            f"{LABEL_MAP.get(r['model'], r['model']):<10} "
            f"{r['params_M']:>7.1f} {r['train_time_min']:>7.1f} {r['train_peak_mem_GB']:>6.1f} "
            f"{r['val_mAP50']:>7.4f} {r['val_mAP50_95']:>7.4f} "
            f"{r['test_mAP50']:>7.4f} {r['test_mAP50_95']:>7.4f} "
            f"{r['preprocess_ms']:>6.2f} {r['inference_ms']:>6.2f} {r['postprocess_ms']:>6.2f}"
        )


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

    # ---- Inference Speed (stacked: pre + inf + post) ----
    ax = axes[1, 1]
    pre = [r["preprocess_ms"] for r in results]
    inf = [r["inference_ms"] for r in results]
    post = [r["postprocess_ms"] for r in results]
    b1 = ax.bar(x, pre, color="#FF9800", edgecolor="white", label="Preprocess")
    b2 = ax.bar(x, inf, bottom=pre, color="#2196F3", edgecolor="white", label="Inference")
    b3 = ax.bar(x, post, bottom=[p + i for p, i in zip(pre, inf)],
                color="#4CAF50", edgecolor="white", label="Postprocess")
    totals = [p + i + po for p, i, po in zip(pre, inf, post)]
    ax.bar_label(b3, labels=[f"{t:.1f}" for t in totals], fontsize=9, padding=2)
    ax.set_title("Inference Speed Breakdown", fontweight="bold")
    ax.set_ylabel("ms / image")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = str(STUDY_DIR / "comparison_charts.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")


# ═══════════════════════════════════════════════════════════
#  Efficiency scatter plots
# ═══════════════════════════════════════════════════════════
def plot_efficiency(results):
    labels = [LABEL_MAP.get(r["model"], r["model"]) for r in results]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#DDA0DD"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("YOLO26 Efficiency Analysis", fontsize=14, fontweight="bold")

    # ---- Accuracy vs Model Size ----
    ax = axes[0]
    for i, r in enumerate(results):
        ax.scatter(r["params_M"], r["test_mAP50_95"], s=180, c=colors[i],
                   edgecolors="black", zorder=3)
        ax.annotate(labels[i], (r["params_M"], r["test_mAP50_95"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Test mAP@50-95", fontsize=12)
    ax.set_title("Accuracy vs Model Size")
    ax.grid(True, alpha=0.3)

    # ---- Accuracy vs Speed ----
    ax = axes[1]
    for i, r in enumerate(results):
        ax.scatter(r["inference_ms"], r["test_mAP50_95"], s=180, c=colors[i],
                   edgecolors="black", zorder=3)
        ax.annotate(labels[i], (r["inference_ms"], r["test_mAP50_95"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
    ax.set_xlabel("Inference Time (ms/image)", fontsize=12)
    ax.set_ylabel("Test mAP@50-95", fontsize=12)
    ax.set_title("Accuracy vs Speed")
    ax.grid(True, alpha=0.3)

    # ---- Accuracy vs VRAM ----
    ax = axes[2]
    for i, r in enumerate(results):
        ax.scatter(r["train_peak_mem_GB"], r["test_mAP50_95"], s=180, c=colors[i],
                   edgecolors="black", zorder=3)
        ax.annotate(labels[i], (r["train_peak_mem_GB"], r["test_mAP50_95"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight="bold")
    ax.set_xlabel("Peak Training VRAM (GB)", fontsize=12)
    ax.set_ylabel("Test mAP@50-95", fontsize=12)
    ax.set_title("Accuracy vs Memory")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = str(STUDY_DIR / "efficiency_plots.png")
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
