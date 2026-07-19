"""
rebuttal_visualization.py
-------------------------
Generate publication-quality figures for ICML rebuttal:
- Runtime comparison table
- Alpha sensitivity curve
- Ablation study comparison

Author: Rebuttal Visualization
Date: 2026-03-25
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pandas as pd

# Set publication quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn')

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color scheme
COLORS = {
    'deterministic': '#2E86AB',    # Blue
    'diagonal': '#A23B72',          # Purple
    'ours': '#F18F01',              # Orange
    'cholesky': '#C73E1D',          # Red
    'direct': '#6A994E',            # Green
    'baseline': '#6C757D'           # Gray
}


def plot_runtime_comparison(json_file='rebuttal_experiment1_runtime.json'):
    """
    Figure 1: Runtime/Overhead Comparison

    Creates a bar chart comparing forward pass time for three models:
    - Deterministic (baseline)
    - Diagonal UQ
    - Ours (Full)
    """
    print("\nGenerating Runtime Comparison Figure...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract data
    models = ['Deterministic', 'Diagonal UQ', 'Ours (Full)']
    times = [data[m]['mean_ms'] for m in models]
    stds = [data[m]['std_ms'] for m in models]
    overhead = data['relative_overhead']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar positions
    x = np.arange(len(models))
    width = 0.6

    # Plot bars
    bars = ax.bar(x, times, width, yerr=stds, capsize=5,
                  color=[COLORS['deterministic'], COLORS['diagonal'], COLORS['ours']],
                  edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.2})

    # Add value labels on bars
    for i, (bar, time, std) in enumerate(zip(bars, times, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{time:.2f}±{std:.2f} ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_ylabel('Forward Pass Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Overhead Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(times) * 1.3)

    # Add grid
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)

    # Add text annotation for relative overhead
    text_str = f"Relative Overhead:\n"
    text_str += f"Diagonal vs Det: {overhead['diagonal_vs_deterministic']:.1f}%\n"
    text_str += f"Full vs Det: {overhead['full_vs_deterministic']:.1f}%\n"
    text_str += f"Full vs Diagonal: {overhead['full_vs_diagonal']:.1f}%"

    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('rebuttal_figure1_runtime.pdf')
    plt.savefig('rebuttal_figure1_runtime.png')
    print("[+] Saved: rebuttal_figure1_runtime.pdf/png")

    return fig, ax


def plot_alpha_sensitivity(json_file='rebuttal_experiment2_alpha_sensitivity.json'):
    """
    Figure 2: Alpha Sensitivity Analysis

    Creates a line plot showing how MAE and NLL change with different alpha values
    """
    print("\nGenerating Alpha Sensitivity Figure...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract data
    alphas = []
    final_maes = []
    final_nlls = []
    best_maes = []
    best_nlls = []

    for key in sorted(data.keys()):
        alphas.append(data[key]['alpha'])
        final_maes.append(data[key]['final_mae'])
        final_nlls.append(data[key]['final_nll'])
        best_maes.append(data[key]['best_mae'])
        best_nlls.append(data[key]['best_nll'])

    alphas = np.array(alphas)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot MAE
    ax1.plot(alphas, final_maes, 'o-', color=COLORS['ours'],
             linewidth=2, markersize=8, label='Final MAE', markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(alphas, best_maes, 's--', color=COLORS['deterministic'],
             linewidth=2, markersize=8, label='Best MAE', markeredgecolor='black', markeredgewidth=1.5)

    ax1.set_xlabel('α (Regularization Strength)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Prediction Accuracy', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xscale('log')
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.legend(loc='best', frameon=True, shadow=True)

    # Add annotation for default value
    default_alpha = 0.1
    if default_alpha in alphas:
        idx = np.where(alphas == default_alpha)[0][0]
        ax1.axvline(default_alpha, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(default_alpha, ax1.get_ylim()[1]*0.95, ' Default',
                fontsize=9, color='red', ha='left', va='top')

    # Plot NLL
    ax2.plot(alphas, final_nlls, 'o-', color=COLORS['ours'],
             linewidth=2, markersize=8, label='Final NLL', markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(alphas, best_nlls, 's--', color=COLORS['deterministic'],
             linewidth=2, markersize=8, label='Best NLL', markeredgecolor='black', markeredgewidth=1.5)

    ax2.set_xlabel('α (Regularization Strength)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NLL', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Uncertainty Quality', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xscale('log')
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='best', frameon=True, shadow=True)

    # Add annotation for default value
    if default_alpha in alphas:
        idx = np.where(alphas == default_alpha)[0][0]
        ax2.axvline(default_alpha, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax2.text(default_alpha, ax2.get_ylim()[1]*0.95, ' Default',
                fontsize=9, color='red', ha='left', va='top')

    plt.tight_layout()
    plt.savefig('rebuttal_figure2_alpha_sensitivity.pdf')
    plt.savefig('rebuttal_figure2_alpha_sensitivity.png')
    print("[+] Saved: rebuttal_figure2_alpha_sensitivity.pdf/png")

    return fig, (ax1, ax2)


def plot_ablation_study(json_file='rebuttal_experiment3_ablation.json'):
    """
    Figure 3: Ablation Study Comparison

    Creates a grouped bar chart comparing different variants:
    - Cholesky (SPD only)
    - Direct (Equivariant only)
    - Ours (Full)
    """
    print("\nGenerating Ablation Study Figure...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract data
    variants = list(data.keys())
    maes = [data[v]['mae_mean'] for v in variants]
    mae_stds = [data[v]['mae_std'] for v in variants]
    nlls = [data[v]['nll_mean'] for v in variants]
    nll_stds = [data[v]['nll_std'] for v in variants]
    spd_rates = [data[v]['spd_rate'] for v in variants]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot MAE and NLL
    x = np.arange(len(variants))
    width = 0.35

    # MAE bars
    bars1 = ax1.bar(x - width/2, maes, width, yerr=mae_stds, capsize=5,
                    label='MAE', color=COLORS['ours'],
                    edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.2})

    # NLL bars (scaled for visibility)
    scale_factor = max(maes) / max(nlls) if max(nlls) > 0 else 1
    nlls_scaled = [n * scale_factor for n in nlls]
    nll_stds_scaled = [s * scale_factor for s in nll_stds]

    bars2 = ax1.bar(x + width/2, nlls_scaled, width, yerr=nll_stds_scaled, capsize=5,
                    label='NLL (scaled)', color=COLORS['diagonal'],
                    edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.2})

    # Add value labels
    for bar, val, std in zip(bars1, maes, mae_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    for bar, val, std in zip(bars2, nlls_scaled, nll_stds_scaled):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{val/scale_factor:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Performance Comparison', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace('A. ', '').replace('B. ', '').replace('C. ', '')
                        for v in variants], fontsize=10, rotation=15, ha='right')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.3)

    # Plot SPD Rate
    bars3 = ax2.bar(x, spd_rates, width, capsize=5,
                    color=[COLORS['cholesky'], COLORS['direct'], COLORS['ours']],
                    edgecolor='black', linewidth=1.2)

    # Add percentage labels
    for bar, rate in zip(bars3, spd_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('SPD Satisfaction Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) SPD Guarantee', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.replace('A. ', '').replace('B. ', '').replace('C. ', '')
                        for v in variants], fontsize=10, rotation=15, ha='right')
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.3)

    # Add 100% reference line
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='100% SPD')
    ax2.legend(loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig('rebuttal_figure3_ablation.pdf')
    plt.savefig('rebuttal_figure3_ablation.png')
    print("[+] Saved: rebuttal_figure3_ablation.pdf/png")

    return fig, (ax1, ax2)


def generate_summary_tables():
    """
    Generate LaTeX tables for the rebuttal
    """
    print("\nGenerating LaTeX Tables...")

    # Table 1: Runtime Analysis
    print("\n" + "="*70)
    print("TABLE 1: Runtime Overhead Analysis")
    print("="*70)

    try:
        with open('rebuttal_experiment1_runtime.json', 'r') as f:
            data = json.load(f)

        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Runtime Overhead Analysis}")
        print("\\label{tab:runtime}")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print("Model & Forward (ms) & Std (ms) & Overhead \\\\")
        print("\\hline")

        for model in ['Deterministic', 'Diagonal UQ', 'Ours (Full)']:
            m = data[model]
            if model == 'Deterministic':
                overhead = "—"
            elif model == 'Diagonal UQ':
                overhead = f"{data['relative_overhead']['diagonal_vs_deterministic']:.1f}\\%"
            else:
                overhead = f"{data['relative_overhead']['full_vs_deterministic']:.1f}\\%"

            print(f"{model} & {m['mean_ms']:.2f} & {m['std_ms']:.2f} & {overhead} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    except FileNotFoundError:
        print("[!] Runtime data not found. Run experiments first.")

    # Table 2: Alpha Sensitivity
    print("\n" + "="*70)
    print("TABLE 2: Alpha Sensitivity Analysis")
    print("="*70)

    try:
        with open('rebuttal_experiment2_alpha_sensitivity.json', 'r') as f:
            data = json.load(f)

        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Alpha Sensitivity Analysis}")
        print("\\label{tab:alpha}")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("$\\alpha$ & Final MAE & Final NLL & Best MAE & Best NLL \\\\")
        print("\\hline")

        for key in sorted(data.keys()):
            r = data[key]
            print(f"{r['alpha']:.2f} & {r['final_mae']:.4f} & {r['final_nll']:.4f} & "
                  f"{r['best_mae']:.4f} & {r['best_nll']:.4f} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    except FileNotFoundError:
        print("[!] Alpha sensitivity data not found. Run experiments first.")

    # Table 3: Ablation Study
    print("\n" + "="*70)
    print("TABLE 3: Ablation Study (SPD + Equivariance)")
    print("="*70)

    try:
        with open('rebuttal_experiment3_ablation.json', 'r') as f:
            data = json.load(f)

        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Ablation Study: SPD and Equivariance}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print("Variant & MAE & NLL & SPD Rate \\\\")
        print("\\hline")

        for variant, metrics in data.items():
            clean_name = variant.replace('A. ', '').replace('B. ', '').replace('C. ', '')
            print(f"{clean_name} & {metrics['mae_mean']:.4f}$\\pm${metrics['mae_std']:.4f} & "
                  f"{metrics['nll_mean']:.4f}$\\pm${metrics['nll_std']:.4f} & "
                  f"{metrics['spd_rate']:.1f}\\% \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    except FileNotFoundError:
        print("[!] Ablation data not found. Run experiments first.")


def main():
    """Generate all figures and tables"""
    print("\n" + "="*70)
    print("REBUTTAL VISUALIZATION")
    print("="*70)

    # Check if experiment data exists
    import os

    has_runtime = os.path.exists('rebuttal_experiment1_runtime.json')
    has_alpha = os.path.exists('rebuttal_experiment2_alpha_sensitivity.json')
    has_ablation = os.path.exists('rebuttal_experiment3_ablation.json')

    if not (has_runtime or has_alpha or has_ablation):
        print("\n[!] No experiment data found!")
        print("Please run rebuttal_experiments.py first to generate the data.")
        return

    # Generate figures
    if has_runtime:
        try:
            plot_runtime_comparison()
            print("[OK] Runtime figure generated")
        except Exception as e:
            print(f"[FAIL] Runtime figure failed: {e}")

    if has_alpha:
        try:
            plot_alpha_sensitivity()
            print("[OK] Alpha sensitivity figure generated")
        except Exception as e:
            print(f"[FAIL] Alpha sensitivity figure failed: {e}")

    if has_ablation:
        try:
            plot_ablation_study()
            print("[OK] Ablation figure generated")
        except Exception as e:
            print(f"[FAIL] Ablation figure failed: {e}")

    # Generate LaTeX tables
    print("\n" + "="*70)
    print("LATEX TABLES")
    print("="*70)
    generate_summary_tables()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - rebuttal_figure1_runtime.pdf/png")
    print("  - rebuttal_figure2_alpha_sensitivity.pdf/png")
    print("  - rebuttal_figure3_ablation.pdf/png")
    print("\nLaTeX tables have been printed above.")
    print("You can copy them directly to your rebuttal document.")


if __name__ == "__main__":
    main()
