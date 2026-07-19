"""
Risk-Coverage Analysis for ICML Rebuttal
==========================================

This file implements the risk-coverage comparison experiments to fairly evaluate
the contribution of λmax ranking vs other ranking metrics.

Priority 2: λmax Contribution Analysis
-------------------------------------
4. Risk-coverage comparison across baselines
   - Deterministic: ensemble variance / residual proxy ranking
   - Diagonal UQ: max diagonal variance / trace ranking
   - Full covariance: trace / λmax ranking

This addresses the reviewer question:
"Is the 3.1% improvement from the method itself, or from the ranking metric?"

Author: Rebuttal Experiments
Date: 2026-03-29
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from equivariant_network import EquivariantUncertaintyNetwork
from voigt_utils import kelvin_mandel_to_voigt, voigt_to_kelvin_mandel
from stable_loss_implementation import safe_eigh


# ============================================================
# RANKING METRICS
# ============================================================

class RankingMetric:
    """Base class for ranking metrics"""
    def __init__(self, name: str):
        self.name = name

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty/risk score for each sample.

        Args:
            mu_km: [B, 6] predictions in Kelvin-Mandel space
            Sigma_km: [B, 6, 6] covariance matrices (None for deterministic)
            targets_km: [B, 6] ground truth

        Returns:
            scores: [B] uncertainty scores (higher = more uncertain)
        """
        raise NotImplementedError


class ResidualErrorRanking(RankingMetric):
    """
    Deterministic: Rank by residual error (L2 norm of prediction error)
    This is a proxy for uncertainty when no UQ is available.
    """
    def __init__(self):
        super().__init__("Residual Error (L2)")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Residual error: ||mu - target||_2
        residual = mu_km - targets_km
        scores = torch.norm(residual, p=2, dim=-1)
        return scores


class EnsembleVarianceRanking(RankingMetric):
    """
    Deterministic: Rank by ensemble variance (if multiple models available)
    For single model, fallback to prediction norm as proxy.
    """
    def __init__(self):
        super().__init__("Ensemble Variance Proxy")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Use prediction norm as variance proxy
        # Larger predictions -> potentially higher uncertainty
        scores = torch.norm(mu_km, p=2, dim=-1)
        return scores


class MaxDiagonalVarianceRanking(RankingMetric):
    """
    Diagonal UQ: Rank by maximum diagonal variance
    This is the most conservative diagonal metric.
    """
    def __init__(self):
        super().__init__("Max Diagonal Variance")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Extract diagonal variances
        diag_vars = torch.diagonal(Sigma_km, dim1=-2, dim2=-1)
        # Take maximum across 6 components
        scores = diag_vars.max(dim=-1)[0]
        return scores


class TraceRanking(RankingMetric):
    """
    Diagonal/Full UQ: Rank by trace of covariance matrix
    This is the total variance.
    """
    def __init__(self):
        super().__init__("Trace (Total Variance)")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Trace: sum of diagonal elements
        scores = torch.diagonal(Sigma_km, dim1=-2, dim2=-1).sum(dim=-1)
        return scores


class LambdaMaxRanking(RankingMetric):
    """
    Full Covariance UQ: Rank by maximum eigenvalue (λmax)
    This is the method used in the paper.
    """
    def __init__(self):
        super().__init__("λmax (Max Eigenvalue)")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Compute eigenvalues and take maximum
        try:
            eigenvals = torch.linalg.eigvalsh(Sigma_km)
            scores = eigenvals[:, -1]  # Maximum eigenvalue
        except:
            # Fallback to trace if eigenvalue decomposition fails
            scores = torch.diagonal(Sigma_km, dim1=-2, dim2=-1).sum(dim=-1)
        return scores


class MahalanobisDistanceRanking(RankingMetric):
    """
    Full Covariance UQ: Rank by Mahalanobis distance
    This accounts for full covariance structure.
    """
    def __init__(self):
        super().__init__("Mahalanobis Distance")

    def compute_score(self, mu_km: torch.Tensor, Sigma_km: torch.Tensor,
                     targets_km: torch.Tensor) -> torch.Tensor:
        # Mahalanobis distance: (mu - target)^T @ Sigma^{-1} @ (mu - target)
        diff = (mu_km - targets_km).unsqueeze(-1)
        try:
            inv_sigma_diff = torch.linalg.solve(Sigma_km, diff)
            scores = torch.bmm(diff.transpose(1, 2), inv_sigma_diff).squeeze(-1).squeeze(-1)
            scores = torch.sqrt(scores)  # Convert to distance
        except:
            # Fallback to Euclidean distance
            scores = torch.norm(mu_km - targets_km, p=2, dim=-1)
        return scores


# ============================================================
# RISK-COVERAGE COMPUTATION
# ============================================================

def compute_risk_coverage(scores: np.ndarray, errors: np.ndarray,
                          num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve.

    Args:
        scores: [N] uncertainty scores (higher = more uncertain)
        errors: [N] prediction errors (e.g., MAE)
        num_points: Number of points on the curve

    Returns:
        coverage: [num_points] fraction of samples kept (0 to 1)
        risk: [num_points] average error of kept samples
    """
    # Sort by uncertainty score (ascending = most confident first)
    sorted_indices = np.argsort(scores)
    sorted_errors = errors[sorted_indices]

    # Compute risk at different coverage levels
    coverage_levels = np.linspace(0.01, 1.0, num_points)
    risk_values = []

    for coverage in coverage_levels:
        # Keep top (coverage * 100)% most confident samples
        num_keep = int(coverage * len(sorted_errors))
        if num_keep > 0:
            kept_errors = sorted_errors[:num_keep]
            risk_values.append(kept_errors.mean())
        else:
            risk_values.append(sorted_errors[0])

    return coverage_levels, np.array(risk_values)


def compute_area_under_risk(coverage: np.ndarray, risk: np.ndarray) -> float:
    """
    Compute Area Under Risk (AUR) curve.
    Lower is better (less risk at same coverage).
    """
    # Normalize to [0, 1] range
    coverage_norm = coverage / coverage.max()
    risk_norm = risk / risk.max()

    # Compute AUC using trapezoidal rule
    aur = np.trapz(risk_norm, coverage_norm)
    return aur


def compute_improvement_percentage(baseline_risk: np.ndarray, new_risk: np.ndarray,
                                   coverage: np.ndarray) -> float:
    """
    Compute percentage improvement over baseline.
    """
    # Average improvement across all coverage levels
    improvement = ((baseline_risk - new_risk) / baseline_risk).mean() * 100
    return improvement


# ============================================================
# EXPERIMENT 4: RISK-COVERAGE COMPARISON
# ============================================================

@dataclass
class RiskCoverageResults:
    """Results for risk-coverage experiment"""
    method_name: str
    ranking_metric: str
    coverage: np.ndarray
    risk: np.ndarray
    aur: float
    mae_at_50_coverage: float
    mae_at_90_coverage: float


def risk_coverage_experiment(model, data_loader, device,
                            model_type: str = "full") -> Dict[str, RiskCoverageResults]:
    """
    Experiment 4: Risk-Coverage Comparison

    Tests different ranking metrics for each model type:
    - Deterministic: residual error, ensemble variance proxy
    - Diagonal UQ: max diagonal variance, trace
    - Full Covariance: trace, λmax, Mahalanobis distance

    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to run on
        model_type: 'deterministic', 'diagonal', or 'full'

    Returns:
        Dictionary mapping metric names to RiskCoverageResults
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT 4: RISK-COVERAGE ANALYSIS ({model_type.upper()} Model)")
    print("="*70)

    # Define ranking metrics based on model type
    if model_type == "deterministic":
        ranking_metrics = {
            "residual_error": ResidualErrorRanking(),
            "ensemble_variance": EnsembleVarianceRanking(),
        }
    elif model_type == "diagonal":
        ranking_metrics = {
            "max_diagonal": MaxDiagonalVarianceRanking(),
            "trace": TraceRanking(),
        }
    else:  # full
        ranking_metrics = {
            "trace": TraceRanking(),
            "lambda_max": LambdaMaxRanking(),
            "mahalanobis": MahalanobisDistanceRanking(),
        }

    # Collect all predictions and targets
    all_mu_km = []
    all_Sigma_km = []
    all_targets_km = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            batch = batch.to(device)
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                continue

            try:
                if model_type == "deterministic":
                    mu_km = model(batch)
                    Sigma_km = None
                else:
                    mu_km, _, Sigma_km = model(batch, compute_sigma=True)

                targets_km = batch.y.view(mu_km.shape[0], 6)

                all_mu_km.append(mu_km.cpu())
                all_targets_km.append(targets_km.cpu())
                if Sigma_km is not None:
                    all_Sigma_km.append(Sigma_km.cpu())

            except Exception as e:
                print(f"Warning: {e}")
                continue

    # Concatenate all batches
    all_mu_km = torch.cat(all_mu_km, dim=0).numpy()
    all_targets_km = torch.cat(all_targets_km, dim=0).numpy()
    if all_Sigma_km:
        all_Sigma_km = torch.cat(all_Sigma_km, dim=0)

    # Compute prediction errors
    errors = np.abs(all_mu_km - all_targets_km).mean(axis=1)  # Average error across 6 components

    results = {}

    # Test each ranking metric
    for metric_key, ranking_metric in ranking_metrics.items():
        print(f"\nTesting ranking metric: {ranking_metric.name}")

        # Compute uncertainty scores
        if model_type == "deterministic":
            mu_tensor = torch.from_numpy(all_mu_km).to(device)
            target_tensor = torch.from_numpy(all_targets_km).to(device)
            scores = ranking_metric.compute_score(mu_tensor, None, target_tensor)
        else:
            mu_tensor = torch.from_numpy(all_mu_km).to(device)
            sigma_tensor = torch.from_numpy(all_Sigma_km).to(device)
            target_tensor = torch.from_numpy(all_targets_km).to(device)
            scores = ranking_metric.compute_score(mu_tensor, sigma_tensor, target_tensor)

        scores_np = scores.cpu().numpy()

        # Compute risk-coverage curve
        coverage, risk = compute_risk_coverage(scores_np, errors)

        # Compute metrics
        aur = compute_area_under_risk(coverage, risk)

        # Find MAE at specific coverage levels
        idx_50 = np.argmin(np.abs(coverage - 0.50))
        idx_90 = np.argmin(np.abs(coverage - 0.90))

        mae_at_50 = risk[idx_50]
        mae_at_90 = risk[idx_90]

        results[metric_key] = RiskCoverageResults(
            method_name=model_type,
            ranking_metric=ranking_metric.name,
            coverage=coverage,
            risk=risk,
            aur=aur,
            mae_at_50_coverage=mae_at_50,
            mae_at_90_coverage=mae_at_90
        )

        print(f"  AUR: {aur:.4f}")
        print(f"  MAE at 50% coverage: {mae_at_50:.4f}")
        print(f"  MAE at 90% coverage: {mae_at_90:.4f}")

    return results


def compare_all_methods(full_model, diagonal_model, deterministic_model,
                       data_loader, device) -> Dict[str, Dict]:
    """
    Compare risk-coverage across all model types and ranking metrics.

    This produces the "fair comparison" that reviewers requested:
    - Compare different ranking metrics within each model type
    - Compare different model types with their best metrics
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE RISK-COVERAGE COMPARISON")
    print("="*70)

    all_results = {}

    # Test full covariance model
    print("\n" + "-"*70)
    print("Testing Full Covariance Model")
    print("-"*70)
    full_results = risk_coverage_experiment(full_model, data_loader, device, model_type="full")
    all_results['full'] = {k: v.__dict__ for k, v in full_results.items()}

    # Test diagonal UQ model
    print("\n" + "-"*70)
    print("Testing Diagonal UQ Model")
    print("-"*70)
    diag_results = risk_coverage_experiment(diagonal_model, data_loader, device, model_type="diagonal")
    all_results['diagonal'] = {k: v.__dict__ for k, v in diag_results.items()}

    # Test deterministic model
    print("\n" + "-"*70)
    print("Testing Deterministic Model")
    print("-"*70)
    det_results = risk_coverage_experiment(deterministic_model, data_loader, device, model_type="deterministic")
    all_results['deterministic'] = {k: v.__dict__ for k, v in det_results.items()}

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: BEST RANKING METRIC FOR EACH METHOD")
    print("="*70)

    best_results = {}

    for model_type, model_results in all_results.items():
        # Find best metric (lowest AUR)
        best_metric = min(model_results.items(), key=lambda x: x[1]['aur'])
        best_results[model_type] = best_metric

        print(f"\n{model_type.upper()}:")
        print(f"  Best Metric: {best_metric[1]['ranking_metric']}")
        print(f"  AUR: {best_metric[1]['aur']:.4f}")
        print(f"  MAE at 50% coverage: {best_metric[1']['mae_at_50_coverage']:.4f}")
        print(f"  MAE at 90% coverage: {best_metric[1]['mae_at_90_coverage']:.4f}")

    # Compute improvement percentages
    print("\n" + "-"*70)
    print("IMPROVEMENT ANALYSIS")
    print("-"*70)

    baseline_aur = best_results['deterministic'][1]['aur']
    diagonal_aur = best_results['diagonal'][1]['aur']
    full_aur = best_results['full'][1]['aur']

    diag_improvement = ((baseline_aur - diagonal_aur) / baseline_aur) * 100
    full_improvement = ((baseline_aur - full_aur) / baseline_aur) * 100
    full_vs_diag = ((diagonal_aur - full_aur) / diagonal_aur) * 100

    print(f"Diagonal UQ vs Deterministic: {diag_improvement:.2f}% improvement")
    print(f"Full UQ vs Deterministic: {full_improvement:.2f}% improvement")
    print(f"Full UQ vs Diagonal UQ: {full_vs_diag:.2f}% improvement")

    # Save results
    output_file = 'rebuttal_exp4_risk_coverage.json'
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_type, model_results in all_results.items():
            serializable_results[model_type] = {}
            for metric_key, metric_results in model_results.items():
                serializable_results[model_type][metric_key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in metric_results.items()
                }
        json.dump(serializable_results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return all_results, best_results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_risk_coverage_curves(all_results: Dict, save_path: str = 'rebuttal_risk_coverage_curves.png'):
    """
    Create publication-quality risk-coverage curves.

    Shows:
    1. Different ranking metrics within each model type (subplot)
    2. Best metrics across all model types (main comparison)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-.']

    # Plot 1: Deterministic methods
    ax = axes[0]
    det_results = all_results.get('deterministic', {})
    for i, (metric_key, results) in enumerate(det_results.items()):
        ax.plot(results['coverage'], results['risk'],
                label=results['ranking_metric'],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2)
    ax.set_xlabel('Coverage (Fraction of Samples)', fontsize=12)
    ax.set_ylabel('Risk (Mean Absolute Error)', fontsize=12)
    ax.set_title('(a) Deterministic Baselines', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # Plot 2: Diagonal UQ methods
    ax = axes[1]
    diag_results = all_results.get('diagonal', {})
    for i, (metric_key, results) in enumerate(diag_results.items()):
        ax.plot(results['coverage'], results['risk'],
                label=results['ranking_metric'],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2)
    ax.set_xlabel('Coverage (Fraction of Samples)', fontsize=12)
    ax.set_ylabel('Risk (Mean Absolute Error)', fontsize=12)
    ax.set_title('(b) Diagonal UQ Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # Plot 3: Full covariance methods
    ax = axes[2]
    full_results = all_results.get('full', {})
    for i, (metric_key, results) in enumerate(full_results.items()):
        ax.plot(results['coverage'], results['risk'],
                label=results['ranking_metric'],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2)
    ax.set_xlabel('Coverage (Fraction of Samples)', fontsize=12)
    ax.set_ylabel('Risk (Mean Absolute Error)', fontsize=12)
    ax.set_title('(c) Full Covariance Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to {save_path}")

    # Also create a combined plot showing best metrics
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))

    # Get best metric from each model type
    best_metrics = {}
    for model_type, model_results in all_results.items():
        best = min(model_results.items(), key=lambda x: x[1]['aur'])
        best_metrics[model_type] = best

    # Plot best metrics
    model_names = {
        'deterministic': 'Deterministic (Best)',
        'diagonal': 'Diagonal UQ (Best)',
        'full': 'Full Covariance (Best)'
    }

    colors_combined = ['#d62728', '#ff7f0e', '#1f77b4']

    for i, (model_type, (metric_key, results)) in enumerate(best_metrics.items()):
        ax2.plot(results['coverage'], results['risk'],
                label=model_names[model_type],
                color=colors_combined[i],
                linewidth=3,
                marker='o',
                markersize=4,
                markevery=10)

    ax2.set_xlabel('Coverage (Fraction of Samples)', fontsize=14)
    ax2.set_ylabel('Risk (Mean Absolute Error)', fontsize=14)
    ax2.set_title('Best Risk-Coverage Comparison Across Methods', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    # Add annotation for improvement
    det_aur = best_metrics['deterministic'][1]['aur']
    full_aur = best_metrics['full'][1]['aur']
    improvement = ((det_aur - full_aur) / det_aur) * 100

    ax2.text(0.5, 0.5, f'Full UQ Improvement: {improvement:.1f}%',
             transform=ax2.transAxes,
             fontsize=12,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    combined_save_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Combined plot saved to {combined_save_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_risk_coverage_analysis(full_model, diagonal_model, deterministic_model,
                               data_loader, device):
    """
    Run comprehensive risk-coverage analysis for rebuttal.

    This addresses the reviewer question:
    "Is the 3.1% improvement from the method itself, or from the ranking metric?"

    By comparing:
    1. Different ranking metrics within each method
    2. Best metrics across all methods
    3. Both risk-coverage curves and improvement percentages
    """
    print("\n" + "="*70)
    print("RISK-COVERAGE ANALYSIS FOR REBUTTAL")
    print("="*70)

    # Run comprehensive comparison
    all_results, best_results = compare_all_methods(
        full_model, diagonal_model, deterministic_model, data_loader, device
    )

    # Generate plots
    plot_risk_coverage_curves(all_results)

    return all_results, best_results


if __name__ == "__main__":
    print("""
    ====================================
    Risk-Coverage Analysis for Rebuttal
    ====================================

    To run these experiments, you need to:
    1. Have trained models (full, diagonal, deterministic)
    2. Set up the data loader
    3. Call run_risk_coverage_analysis(full_model, diagonal_model,
                                       deterministic_model, data_loader, device)

    This will:
    - Test different ranking metrics for each model type
    - Generate risk-coverage curves
    - Compute improvement percentages
    - Create publication-quality plots
    """)
