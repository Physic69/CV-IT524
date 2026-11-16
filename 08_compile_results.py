#!/usr/bin/env python3
"""
08_compile_results.py
=====================
Load all saved eval results from scripts 02-07, build the full results table,
compute domain gaps, and generate comparison visualizations.

This is the FINAL script. It produces the headline findings:
  - Full table: model × train_domain × test_domain × per-class IoU × mIoU
  - Domain gap: in-domain mIoU minus cross-domain mIoU for each condition
  - Bar charts comparing SAM2 vs DeepLabV3 domain gaps
  - Side-by-side qualitative comparison grids

Run:   python 08_compile_results.py
Outputs: ./outputs/08_results/
"""

import os, sys, json, glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./outputs/08_results"
DATA_ROOT  = "./data"

# Paths to results from previous scripts
RESULTS_PATHS = {
    "zero_shot":       "./outputs/02_zero_shot/zero_shot_results.json",
    "sam2_urban":      "./outputs/04_sam2_urban/results.json",
    "sam2_rural":      "./outputs/05_sam2_rural/results.json",
    "baseline_urban":  "./outputs/06_baseline_urban/results.json",
    "baseline_rural":  "./outputs/07_baseline_rural/results.json",
}

CLASS_NAMES = ["no-data","background","building","road","water","barren","forest","agriculture"]
NUM_CLASSES = 7

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
def load_results():
    results = {}
    for key, path in RESULTS_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
            print(f"  [✓] Loaded {key}: {path}")
        else:
            print(f"  [!] Missing {key}: {path}")
            results[key] = None
    return results

def get_final_eval(result_json):
    """Extract the final epoch's evaluation results."""
    if result_json is None:
        return None, None
    evals = result_json.get("eval_results", {})
    if not evals:
        return None, None
    # Get the last epoch key
    last_key = sorted(evals.keys(), key=lambda x: int(x.split("_")[1]))[-1]
    return evals[last_key].get("in_domain"), evals[last_key].get("cross_domain")

# ---------------------------------------------------------------------------
# Build master table
# ---------------------------------------------------------------------------
def build_master_table(results):
    """
    Build the full experimental matrix table.

    Rows:
      1. Zero-shot SAM2 (AMG)          — Urban, Rural
      2. Zero-shot SAM2 (Oracle)       — Urban, Rural
      3. SAM2 fine-tuned on Urban      — Urban (in), Rural (cross)
      4. SAM2 fine-tuned on Rural      — Rural (in), Urban (cross)
      5. DeepLabV3 trained on Urban    — Urban (in), Rural (cross)
      6. DeepLabV3 trained on Rural    — Rural (in), Urban (cross)

    Columns: per-class IoU (7 classes) + mIoU
    """
    table = []

    # --- Zero-shot ---
    zs = results.get("zero_shot")
    if zs:
        for strategy, prefix in [("AMG", "amg"), ("Oracle", "oracle")]:
            for scene in ["urban", "rural"]:
                key = f"{prefix}_{scene}"
                if key in zs:
                    entry = zs[key]
                    table.append({
                        "model": f"SAM2 Zero-shot ({strategy})",
                        "train_domain": "None",
                        "test_domain": scene.capitalize(),
                        "per_class": entry["per_class_iou"],
                        "miou": entry["mean_iou"],
                    })

    # --- SAM2 fine-tuned ---
    for run_key, train_scene, cross_scene in [
        ("sam2_urban", "Urban", "Rural"),
        ("sam2_rural", "Rural", "Urban"),
    ]:
        in_dom, cross_dom = get_final_eval(results.get(run_key))
        if in_dom:
            table.append({
                "model": "SAM2 Fine-tuned",
                "train_domain": train_scene,
                "test_domain": train_scene,
                "type": "in-domain",
                "per_class": in_dom["per_class"],
                "miou": in_dom["miou"],
            })
        if cross_dom:
            table.append({
                "model": "SAM2 Fine-tuned",
                "train_domain": train_scene,
                "test_domain": cross_scene,
                "type": "cross-domain",
                "per_class": cross_dom["per_class"],
                "miou": cross_dom["miou"],
            })

    # --- DeepLabV3 ---
    for run_key, train_scene, cross_scene in [
        ("baseline_urban", "Urban", "Rural"),
        ("baseline_rural", "Rural", "Urban"),
    ]:
        in_dom, cross_dom = get_final_eval(results.get(run_key))
        if in_dom:
            table.append({
                "model": "DeepLabV3-ResNet50",
                "train_domain": train_scene,
                "test_domain": train_scene,
                "type": "in-domain",
                "per_class": in_dom["per_class"],
                "miou": in_dom["miou"],
            })
        if cross_dom:
            table.append({
                "model": "DeepLabV3-ResNet50",
                "train_domain": train_scene,
                "test_domain": cross_scene,
                "type": "cross-domain",
                "per_class": cross_dom["per_class"],
                "miou": cross_dom["miou"],
            })

    return table

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
def print_table(table):
    print("\n" + "=" * 120)
    print("  FULL RESULTS TABLE")
    print("=" * 120)

    header = f"{'Model':<28s} {'Train':<8s} {'Test':<8s}"
    for c in range(1, NUM_CLASSES + 1):
        header += f" {CLASS_NAMES[c]:>10s}"
    header += f" {'mIoU':>8s}"
    print(header)
    print("-" * 120)

    for row in table:
        line = f"{row['model']:<28s} {row['train_domain']:<8s} {row['test_domain']:<8s}"
        pc = row['per_class']
        for c in range(1, NUM_CLASSES + 1):
            val = pc.get(str(c), float('nan'))
            if isinstance(val, str):
                val = float(val) if val != 'nan' else float('nan')
            if np.isnan(val):
                line += f" {'N/A':>10s}"
            else:
                line += f" {val*100:>9.2f}%"
        line += f" {row['miou']*100:>7.2f}%"
        print(line)

    print("=" * 120)

# ---------------------------------------------------------------------------
# Domain gap analysis
# ---------------------------------------------------------------------------
def compute_domain_gaps(table):
    """
    Compute the HEADLINE metric: in-domain mIoU minus cross-domain mIoU.
    A smaller gap means better cross-domain generalization.
    """
    gaps = []
    print("\n" + "=" * 80)
    print("  DOMAIN GAP ANALYSIS  (in-domain mIoU − cross-domain mIoU)")
    print("  Smaller gap = better cross-domain generalization")
    print("=" * 80)

    # Group by model + train_domain
    from collections import defaultdict
    groups = defaultdict(dict)
    for row in table:
        if row.get('type') in ['in-domain', 'cross-domain']:
            key = (row['model'], row['train_domain'])
            groups[key][row['type']] = row['miou']

    print(f"\n  {'Model':<28s} {'Train→':<10s} {'In-dom':>8s} {'Cross':>8s} {'Gap':>8s}")
    print("  " + "-" * 70)

    for (model, train_dom), vals in sorted(groups.items()):
        in_miou = vals.get('in-domain', float('nan'))
        cross_miou = vals.get('cross-domain', float('nan'))
        gap = in_miou - cross_miou if not (np.isnan(in_miou) or np.isnan(cross_miou)) else float('nan')
        gaps.append({
            "model": model, "train_domain": train_dom,
            "in_domain_miou": in_miou, "cross_domain_miou": cross_miou,
            "gap": gap,
        })
        in_str = f"{in_miou*100:.2f}%" if not np.isnan(in_miou) else "N/A"
        cross_str = f"{cross_miou*100:.2f}%" if not np.isnan(cross_miou) else "N/A"
        gap_str = f"{gap*100:.2f}%" if not np.isnan(gap) else "N/A"
        print(f"  {model:<28s} {train_dom:<10s} {in_str:>8s} {cross_str:>8s} {gap_str:>8s}")

    print()
    return gaps

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_domain_gap_comparison(gaps, output_dir):
    """Bar chart: SAM2 gap vs DeepLabV3 gap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    gap_vals = []
    colors = []
    labels_list = []

    for g in gaps:
        if np.isnan(g['gap']):
            continue
        label = f"{g['model']}\n(Train: {g['train_domain']})"
        models.append(label)
        gap_vals.append(g['gap'] * 100)
        colors.append('#3b82f6' if 'SAM2' in g['model'] else '#ef4444')
        labels_list.append(g['model'])

    if not models:
        print("  [!] No gap data to plot")
        return

    x = np.arange(len(models))
    bars = ax.bar(x, gap_vals, color=colors, edgecolor='white', linewidth=1.5, width=0.5)

    for bar, val in zip(bars, gap_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Domain Gap (mIoU %)", fontsize=12)
    ax.set_title("Cross-Domain Generalization Gap\n(In-domain mIoU − Cross-domain mIoU)\nSmaller = Better Generalization",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Legend
    sam_patch = mpatches.Patch(color='#3b82f6', label='SAM2 Fine-tuned')
    dlv3_patch = mpatches.Patch(color='#ef4444', label='DeepLabV3-ResNet50')
    ax.legend(handles=[sam_patch, dlv3_patch], loc='upper right', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "domain_gap_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Domain gap chart saved to {path}")


def plot_per_class_comparison(table, output_dir):
    """Grouped bar chart: per-class IoU for SAM2 vs DeepLabV3, in-domain vs cross-domain."""
    # Find relevant rows
    sam2_rows = [r for r in table if 'SAM2 Fine' in r['model'] and r.get('type')]
    dlv3_rows = [r for r in table if 'DeepLabV3' in r['model'] and r.get('type')]

    if not sam2_rows or not dlv3_rows:
        print("  [!] Not enough data for per-class comparison")
        return

    for direction in ["Urban", "Rural"]:
        fig, ax = plt.subplots(figsize=(14, 6))

        sam2_in = next((r for r in sam2_rows if r['train_domain']==direction and r['type']=='in-domain'), None)
        sam2_cross = next((r for r in sam2_rows if r['train_domain']==direction and r['type']=='cross-domain'), None)
        dlv3_in = next((r for r in dlv3_rows if r['train_domain']==direction and r['type']=='in-domain'), None)
        dlv3_cross = next((r for r in dlv3_rows if r['train_domain']==direction and r['type']=='cross-domain'), None)

        classes = [CLASS_NAMES[c] for c in range(1, NUM_CLASSES+1)]
        x = np.arange(len(classes))
        width = 0.2

        def get_vals(row):
            if row is None:
                return [0]*NUM_CLASSES
            pc = row['per_class']
            return [float(pc.get(str(c), 0))*100 for c in range(1, NUM_CLASSES+1)]

        if sam2_in:
            ax.bar(x - 1.5*width, get_vals(sam2_in), width, label='SAM2 In-domain', color='#3b82f6')
        if sam2_cross:
            ax.bar(x - 0.5*width, get_vals(sam2_cross), width, label='SAM2 Cross-domain', color='#93c5fd')
        if dlv3_in:
            ax.bar(x + 0.5*width, get_vals(dlv3_in), width, label='DeepLabV3 In-domain', color='#ef4444')
        if dlv3_cross:
            ax.bar(x + 1.5*width, get_vals(dlv3_cross), width, label='DeepLabV3 Cross-domain', color='#fca5a5')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha='right')
        ax.set_ylabel("IoU (%)")
        ax.set_title(f"Per-Class IoU Comparison (Trained on {direction})", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"per_class_comparison_train_{direction.lower()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [✓] Per-class chart saved to {path}")


def plot_loss_curves_combined(output_dir):
    """Overlay all training loss curves in one plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    curves = [
        ("./outputs/04_sam2_urban/results.json", "SAM2 Urban", '#3b82f6', '-'),
        ("./outputs/05_sam2_rural/results.json", "SAM2 Rural", '#22c55e', '-'),
        ("./outputs/06_baseline_urban/results.json", "DeepLabV3 Urban", '#ef4444', '--'),
        ("./outputs/07_baseline_rural/results.json", "DeepLabV3 Rural", '#f97316', '--'),
    ]

    any_plotted = False
    for path, label, color, style in curves:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            loss = data.get("loss_history", [])
            if loss:
                ax.plot(range(1, len(loss)+1), loss, style, color=color,
                        label=label, linewidth=2, markersize=3)
                any_plotted = True

    if any_plotted:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves — All Models", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "all_loss_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Combined loss curves saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(table, gaps, output_dir):
    """Save results as CSV files for easy import into papers/slides."""
    # Main table
    csv_path = os.path.join(output_dir, "results_table.csv")
    with open(csv_path, "w") as f:
        header = "Model,Train Domain,Test Domain"
        for c in range(1, NUM_CLASSES + 1):
            header += f",{CLASS_NAMES[c]}_IoU"
        header += ",mIoU\n"
        f.write(header)
        for row in table:
            line = f"{row['model']},{row['train_domain']},{row['test_domain']}"
            pc = row['per_class']
            for c in range(1, NUM_CLASSES + 1):
                val = pc.get(str(c), float('nan'))
                if isinstance(val, str):
                    val = float(val)
                line += f",{val:.4f}" if not np.isnan(val) else ",NaN"
            line += f",{row['miou']:.4f}\n"
            f.write(line)
    print(f"  [✓] CSV table saved to {csv_path}")

    # Domain gaps
    gap_path = os.path.join(output_dir, "domain_gaps.csv")
    with open(gap_path, "w") as f:
        f.write("Model,Train Domain,In-domain mIoU,Cross-domain mIoU,Gap\n")
        for g in gaps:
            f.write(f"{g['model']},{g['train_domain']},"
                    f"{g['in_domain_miou']:.4f},{g['cross_domain_miou']:.4f},"
                    f"{g['gap']:.4f}\n")
    print(f"  [✓] Domain gaps CSV saved to {gap_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Compile All Results")
    print("  Script 08 of 08")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load
    print("\n[1] Loading results from all experiments ...")
    results = load_results()

    # Build table
    print("\n[2] Building master results table ...")
    table = build_master_table(results)

    if not table:
        print("  [!] No results found. Run scripts 02-07 first.")
        sys.exit(1)

    # Print
    print_table(table)

    # Domain gaps
    print("\n[3] Computing domain gaps ...")
    gaps = compute_domain_gaps(table)

    # Visualizations
    print("\n[4] Generating visualizations ...")
    plot_domain_gap_comparison(gaps, OUTPUT_DIR)
    plot_per_class_comparison(table, OUTPUT_DIR)
    plot_loss_curves_combined(OUTPUT_DIR)

    # Save CSV
    print("\n[5] Saving CSV tables ...")
    save_csv(table, gaps, OUTPUT_DIR)

    # Save full JSON
    full_results = {"table": table, "domain_gaps": gaps}
    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"  [✓] Full JSON saved to {OUTPUT_DIR}/full_results.json")

    # Final summary
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)
    print("\n  Key outputs:")
    print(f"    - Results table:  {OUTPUT_DIR}/results_table.csv")
    print(f"    - Domain gaps:    {OUTPUT_DIR}/domain_gaps.csv")
    print(f"    - Gap chart:      {OUTPUT_DIR}/domain_gap_comparison.png")
    print(f"    - Per-class IoU:  {OUTPUT_DIR}/per_class_comparison_*.png")
    print(f"    - Loss curves:    {OUTPUT_DIR}/all_loss_curves.png")
    print(f"\n  The HEADLINE metric is the domain gap:")
    print(f"  If SAM2's gap < DeepLabV3's gap, the hypothesis is supported.")
    print("=" * 80)

if __name__ == "__main__":
    main()
