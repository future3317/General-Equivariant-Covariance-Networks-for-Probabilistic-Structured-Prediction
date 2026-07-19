"""
Lambda Max (λmax) Contribution Analysis for ICML Rebuttal
==========================================================

Focused experiment to answer the reviewer question:
"Is the 3.1% improvement from the method itself, or from the λmax ranking metric?"

This compares:
- Deterministic baseline: residual error ranking
- Diagonal UQ: trace ranking
- Full UQ: trace vs λmax ranking

Author: Rebuttal Experiments
Date: 2026-03-29
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from equivariant_network import EquivariantUncertaintyNetwork
from voigt_utils import kelvin_mandel_to_voigt, voigt_to_kelvin_mandel
from stable_loss_implementation import safe_eigh


# ============================================================
# RANKING METRICS (专注于λmax比较)
# ============================================================

def compute_residual_error_ranking(mu_km: torch.Tensor, targets_km: torch.Tensor) -> torch.Tensor:
    """
    Deterministic: Rank by residual error (L2 norm)
    用于没有UQ的确定性baseline
    """
    residual = mu_km - targets_km
    scores = torch.norm(residual, p=2, dim=-1)
    return scores


def compute_trace_ranking(Sigma_km: torch.Tensor) -> torch.Tensor:
    """
    Diagonal/Full UQ: Rank by trace (total variance)
    这是标准的uncertainty ranking方法
    """
    scores = torch.diagonal(Sigma_km, dim1=-2, dim2=-1).sum(dim=-1)
    return scores


def compute_lambda_max_ranking(Sigma_km: torch.Tensor) -> torch.Tensor:
    """
    Full UQ: Rank by λmax (maximum eigenvalue)
    这是论文中使用的方法
    """
    try:
        eigenvals = torch.linalg.eigvalsh(Sigma_km)
        scores = eigenvals[:, -1]  # Maximum eigenvalue (λmax)
    except:
        # Fallback to trace if eigenvalue decomposition fails
        scores = torch.diagonal(Sigma_km, dim1=-2, dim2=-1).sum(dim=-1)
    return scores


def compute_max_diagonal_ranking(Sigma_km: torch.Tensor) -> torch.Tensor:
    """
    Diagonal UQ: Rank by maximum diagonal variance
    这是最保守的对角线metric
    """
    diag_vars = torch.diagonal(Sigma_km, dim1=-2, dim2=-1)
    scores = diag_vars.max(dim=-1)[0]
    return scores


# ============================================================
# RISK-COVERAGE CURVE计算
# ============================================================

def compute_risk_coverage_curve(scores: np.ndarray, errors: np.ndarray,
                                num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算risk-coverage curve

    Args:
        scores: [N] uncertainty scores (越高=越不确定)
        errors: [N] prediction errors (MAE)
        num_points: curve上的点数

    Returns:
        coverage: [num_points] 保留样本的比例 (0到1)
        risk: [num_points] 保留样本的平均error
    """
    # 按uncertainty score排序（升序=最自信的在前）
    sorted_indices = np.argsort(scores)
    sorted_errors = errors[sorted_indices]

    # 在不同coverage level下计算risk
    coverage_levels = np.linspace(0.05, 1.0, num_points)  # 从5%开始避免极端值
    risk_values = []

    for coverage in coverage_levels:
        # 保留最自信的 (coverage * 100)% 样本
        num_keep = int(coverage * len(sorted_errors))
        if num_keep > 0:
            kept_errors = sorted_errors[:num_keep]
            risk_values.append(kept_errors.mean())
        else:
            risk_values.append(sorted_errors[0])

    return coverage_levels, np.array(risk_values)


def compute_improvement_at_coverage(baseline_risk: np.ndarray, new_risk: np.ndarray,
                                    coverage: np.ndarray, coverage_level: float = 0.90) -> float:
    """
    计算在指定coverage level下的improvement percentage
    """
    idx = np.argmin(np.abs(coverage - coverage_level))
    baseline_error = baseline_risk[idx]
    new_error = new_risk[idx]
    improvement = ((baseline_error - new_error) / baseline_error) * 100
    return improvement


def compute_average_improvement(baseline_risk: np.ndarray, new_risk: np.ndarray) -> Dict[str, float]:
    """
    计算整个curve上的平均improvement
    """
    improvement_per_point = ((baseline_risk - new_risk) / baseline_risk) * 100

    return {
        'mean_improvement': improvement_per_point.mean(),
        'max_improvement': improvement_per_point.max(),
        'min_improvement': improvement_per_point.min(),
        'improvement_at_50': compute_improvement_at_coverage(baseline_risk, new_risk,
                                                              np.linspace(0.05, 1.0, len(baseline_risk)), 0.50),
        'improvement_at_90': compute_improvement_at_coverage(baseline_risk, new_risk,
                                                              np.linspace(0.05, 1.0, len(baseline_risk)), 0.90),
    }


# ============================================================
# 主要实验函数
# ============================================================

def collect_predictions(model, data_loader, device, model_type='full'):
    """
    收集所有predictions和targets
    """
    model.eval()
    all_mu_km = []
    all_Sigma_km = []
    all_targets_km = []

    print(f"Collecting predictions for {model_type} model...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            batch = batch.to(device)
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                continue

            try:
                if model_type == 'deterministic':
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

    # 合并所有batches
    all_mu_km = torch.cat(all_mu_km, dim=0).numpy()
    all_targets_km = torch.cat(all_targets_km, dim=0).numpy()
    if all_Sigma_km:
        all_Sigma_km = torch.cat(all_Sigma_km, dim=0)

    # 计算prediction errors (MAE)
    errors = np.abs(all_mu_km - all_targets_km).mean(axis=1)

    print(f"  Collected {len(all_mu_km)} samples")
    print(f"  Mean MAE: {errors.mean():.4f}")
    print(f"  Std MAE: {errors.std():.4f}")

    return all_mu_km, all_Sigma_km, all_targets_km, errors


def run_lambda_max_comparison(full_model, diagonal_model, deterministic_model,
                              data_loader, device):
    """
    主要实验：λmax vs 其他ranking metrics

    对比：
    1. Deterministic (residual error)
    2. Diagonal UQ (max diagonal variance)
    3. Full UQ with trace ranking
    4. Full UQ with λmax ranking (论文方法)
    """
    print("\n" + "="*70)
    print("LAMBDA MAX (λmax) CONTRIBUTION ANALYSIS")
    print("="*70)

    results = {}

    # ============================================================
    # 1. Deterministic Baseline
    # ============================================================
    print("\n" + "-"*70)
    print("1. DETERMINISTIC BASELINE (Residual Error Ranking)")
    print("-"*70)

    mu_det, _, targets_det, errors_det = collect_predictions(
        deterministic_model, data_loader, device, model_type='deterministic'
    )

    scores_det_residual = compute_residual_error_ranking(
        torch.from_numpy(mu_det).to(device),
        torch.from_numpy(targets_det).to(device)
    ).cpu().numpy()

    coverage_det, risk_det = compute_risk_coverage_curve(scores_det_residual, errors_det)

    results['deterministic'] = {
        'ranking_metric': 'Residual Error',
        'coverage': coverage_det,
        'risk': risk_det,
        'mean_mae': errors_det.mean(),
        'std_mae': errors_det.std(),
    }

    print(f"  Mean MAE: {results['deterministic']['mean_mae']:.4f}")
    print(f"  MAE at 50% coverage: {risk_det[np.argmin(np.abs(coverage_det - 0.50))]:.4f}")
    print(f"  MAE at 90% coverage: {risk_det[np.argmin(np.abs(coverage_det - 0.90))]:.4f}")

    # ============================================================
    # 2. Diagonal UQ
    # ============================================================
    print("\n" + "-"*70)
    print("2. DIAGONAL UQ (Max Diagonal Variance Ranking)")
    print("-"*70)

    mu_diag, Sigma_diag, targets_diag, errors_diag = collect_predictions(
        diagonal_model, data_loader, device, model_type='diagonal'
    )

    scores_diag_max = compute_max_diagonal_ranking(
        torch.from_numpy(Sigma_diag).to(device)
    ).cpu().numpy()

    coverage_diag, risk_diag = compute_risk_coverage_curve(scores_diag_max, errors_diag)

    results['diagonal_uq'] = {
        'ranking_metric': 'Max Diagonal Variance',
        'coverage': coverage_diag,
        'risk': risk_diag,
        'mean_mae': errors_diag.mean(),
        'std_mae': errors_diag.std(),
    }

    print(f"  Mean MAE: {results['diagonal_uq']['mean_mae']:.4f}")
    print(f"  MAE at 50% coverage: {risk_diag[np.argmin(np.abs(coverage_diag - 0.50))]:.4f}")
    print(f"  MAE at 90% coverage: {risk_diag[np.argmin(np.abs(coverage_diag - 0.90))]:.4f}")

    # ============================================================
    # 3. Full UQ with Trace Ranking
    # ============================================================
    print("\n" + "-"*70)
    print("3. FULL UQ (Trace Ranking)")
    print("-"*70)

    mu_full, Sigma_full, targets_full, errors_full = collect_predictions(
        full_model, data_loader, device, model_type='full'
    )

    scores_full_trace = compute_trace_ranking(
        torch.from_numpy(Sigma_full).to(device)
    ).cpu().numpy()

    coverage_full_trace, risk_full_trace = compute_risk_coverage_curve(
        scores_full_trace, errors_full
    )

    results['full_uq_trace'] = {
        'ranking_metric': 'Trace (Total Variance)',
        'coverage': coverage_full_trace,
        'risk': risk_full_trace,
        'mean_mae': errors_full.mean(),
        'std_mae': errors_full.std(),
    }

    print(f"  Mean MAE: {results['full_uq_trace']['mean_mae']:.4f}")
    print(f"  MAE at 50% coverage: {risk_full_trace[np.argmin(np.abs(coverage_full_trace - 0.50))]:.4f}")
    print(f"  MAE at 90% coverage: {risk_full_trace[np.argmin(np.abs(coverage_full_trace - 0.90))]:.4f}")

    # ============================================================
    # 4. Full UQ with Lambda Max Ranking (论文方法)
    # ============================================================
    print("\n" + "-"*70)
    print("4. FULL UQ (λmax Ranking - 论文方法)")
    print("-"*70)

    scores_full_lambda = compute_lambda_max_ranking(
        torch.from_numpy(Sigma_full).to(device)
    ).cpu().numpy()

    coverage_full_lambda, risk_full_lambda = compute_risk_coverage_curve(
        scores_full_lambda, errors_full
    )

    results['full_uq_lambda'] = {
        'ranking_metric': 'λmax (Max Eigenvalue)',
        'coverage': coverage_full_lambda,
        'risk': risk_full_lambda,
        'mean_mae': errors_full.mean(),
        'std_mae': errors_full.std(),
    }

    print(f"  Mean MAE: {results['full_uq_lambda']['mean_mae']:.4f}")
    print(f"  MAE at 50% coverage: {risk_full_lambda[np.argmin(np.abs(coverage_full_lambda - 0.50))]:.4f}")
    print(f"  MAE at 90% coverage: {risk_full_lambda[np.argmin(np.abs(coverage_full_lambda - 0.90))]:.4f}")

    # ============================================================
    # 对比分析
    # ============================================================
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)

    # Full UQ (λmax) vs Deterministic
    improvement_det = compute_average_improvement(risk_det, risk_full_lambda)
    print(f"\nFull UQ (λmax) vs Deterministic:")
    print(f"  Mean improvement: {improvement_det['mean_improvement']:.2f}%")
    print(f"  At 50% coverage: {improvement_det['improvement_at_50']:.2f}%")
    print(f"  At 90% coverage: {improvement_det['improvement_at_90']:.2f}%")

    # Full UQ (λmax) vs Diagonal UQ
    improvement_diag = compute_average_improvement(risk_diag, risk_full_lambda)
    print(f"\nFull UQ (λmax) vs Diagonal UQ:")
    print(f"  Mean improvement: {improvement_diag['mean_improvement']:.2f}%")
    print(f"  At 50% coverage: {improvement_diag['improvement_at_50']:.2f}%")
    print(f"  At 90% coverage: {improvement_diag['improvement_at_90']:.2f}%")

    # Full UQ: λmax vs Trace (关键对比！)
    improvement_trace_vs_lambda = compute_average_improvement(risk_full_trace, risk_full_lambda)
    print(f"\nFull UQ: λmax vs Trace (λmax额外贡献):")
    print(f"  Mean improvement: {improvement_trace_vs_lambda['mean_improvement']:.2f}%")
    print(f"  At 50% coverage: {improvement_trace_vs_lambda['improvement_at_50']:.2f}%")
    print(f"  At 90% coverage: {improvement_trace_vs_lambda['improvement_at_90']:.2f}%")

    # 保存结果
    output_file = 'rebuttal_lambda_max_results.json'
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            'ranking_metric': value['ranking_metric'],
            'mean_mae': float(value['mean_mae']),
            'std_mae': float(value['std_mae']),
            'coverage': value['coverage'].tolist(),
            'risk': value['risk'].tolist(),
        }

    # 添加improvement数据
    serializable_results['improvements'] = {
        'lambda_vs_deterministic': improvement_det,
        'lambda_vs_diagonal': improvement_diag,
        'lambda_vs_trace': improvement_trace_vs_lambda,
    }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results, {
        'lambda_vs_deterministic': improvement_det,
        'lambda_vs_diagonal': improvement_diag,
        'lambda_vs_trace': improvement_trace_vs_lambda,
    }


# ============================================================
# 绘图函数
# ============================================================

def plot_lambda_max_comparison(results, save_path='rebuttal_lambda_max_curves.png'):
    """
    绘制λmax对比图

    创建两张图：
    1. 所有方法的risk-coverage curves
    2. 放大显示Full UQ内部对比 (Trace vs λmax)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1：所有方法对比
    ax1 = axes[0]

    methods = [
        ('deterministic', 'Deterministic\n(Residual Error)', '#d62728', '--'),
        ('diagonal_uq', 'Diagonal UQ\n(Max Diagonal)', '#ff7f0e', '-.'),
        ('full_uq_trace', 'Full UQ\n(Trace)', '#1f77b4', ':'),
        ('full_uq_lambda', 'Full UQ\n(λmax)', '#2ca02c', '-'),
    ]

    for key, label, color, linestyle in methods:
        ax1.plot(results[key]['coverage'], results[key]['risk'],
                label=label, color=color, linestyle=linestyle,
                linewidth=2.5, marker='o', markersize=4, markevery=10)

    ax1.set_xlabel('Coverage (Fraction of Samples Kept)', fontsize=13)
    ax1.set_ylabel('Risk (Mean Absolute Error)', fontsize=13)
    ax1.set_title('(a) All Methods Comparison', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim(bottom=0)

    # 图2：Full UQ内部对比 (Trace vs λmax) - 放大显示差异
    ax2 = axes[1]

    # 只画Full UQ的两个variant
    for key in ['full_uq_trace', 'full_uq_lambda']:
        if key == 'full_uq_trace':
            label = 'Full UQ (Trace)'
            color = '#1f77b4'
            linestyle = ':'
        else:
            label = 'Full UQ (λmax) - Ours'
            color = '#2ca02c'
            linestyle = '-'

        ax2.plot(results[key]['coverage'], results[key]['risk'],
                label=label, color=color, linestyle=linestyle,
                linewidth=3, marker='o', markersize=5, markevery=5)

    ax2.set_xlabel('Coverage (Fraction of Samples Kept)', fontsize=13)
    ax2.set_ylabel('Risk (Mean Absolute Error)', fontsize=13)
    ax2.set_title('(b) λmax vs Trace (Full UQ Internal)', fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim(bottom=0)

    # 添加标注显示λmax的优势
    trace_risk_90 = results['full_uq_trace']['risk'][np.argmin(
        np.abs(results['full_uq_trace']['coverage'] - 0.90)
    )]
    lambda_risk_90 = results['full_uq_lambda']['risk'][np.argmin(
        np.abs(results['full_uq_lambda']['coverage'] - 0.90)
    )]
    improvement = ((trace_risk_90 - lambda_risk_90) / trace_risk_90) * 100

    ax2.annotate(f'λmax improvement at 90% coverage: {improvement:.1f}%',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 fontsize=12, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to {save_path}")

    # 也保存一个单独的Full UQ对比图
    fig2, ax = plt.subplots(1, 1, figsize=(10, 7))

    for key in ['full_uq_trace', 'full_uq_lambda']:
        if key == 'full_uq_trace':
            label = 'Trace Ranking'
            color = '#1f77b4'
            linestyle = ':'
        else:
            label = 'λmax Ranking (Ours)'
            color = '#2ca02c'
            linestyle = '-'

        ax.plot(results[key]['coverage'], results[key]['risk'],
               label=label, color=color, linestyle=linestyle,
               linewidth=3.5, marker='o', markersize=6, markevery=5)

    ax.set_xlabel('Coverage (Fraction of Samples Kept)', fontsize=14)
    ax.set_ylabel('Risk (Mean Absolute Error)', fontsize=14)
    ax.set_title('λmax vs Trace Ranking: Is λmax Necessary?', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)

    # 添加多个coverage level的标注
    for cov_level in [0.50, 0.70, 0.90]:
        idx_trace = np.argmin(np.abs(results['full_uq_trace']['coverage'] - cov_level))
        idx_lambda = np.argmin(np.abs(results['full_uq_lambda']['coverage'] - cov_level))

        risk_trace = results['full_uq_trace']['risk'][idx_trace]
        risk_lambda = results['full_uq_lambda']['risk'][idx_lambda]
        improvement = ((risk_trace - risk_lambda) / risk_trace) * 100

        ax.annotate(f'{cov_level*100:.0f}% coverage: {improvement:.1f}% improvement',
                   xy=(cov_level, risk_lambda),
                   xytext=(cov_level + 0.05, risk_lambda * 0.95),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    save_path_2 = save_path.replace('.png', '_lambda_vs_trace.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    print(f"[OK] Detailed λmax vs Trace plot saved to {save_path_2}")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    运行λmax实验的主函数

    需要提供：
    - full_model: 全协方差UQ模型
    - diagonal_model: 对角UQ模型
    - deterministic_model: 确定性baseline模型
    - data_loader: 验证集数据加载器
    - device: 设备
    """
    print("""
    ====================================
    Lambda Max (λmax) Contribution Analysis
    ====================================

    这个实验回答reviewer的问题：
    "3.1%的改进是来自方法本身，还是ranking metric？"

    对比：
    1. Deterministic baseline (residual error ranking)
    2. Diagonal UQ (max diagonal variance ranking)
    3. Full UQ with trace ranking
    4. Full UQ with λmax ranking (论文方法)

    使用方法：
    -----------
    from rebuttal_lambda_max_analysis import run_lambda_max_comparison

    results, improvements = run_lambda_max_comparison(
        full_model, diagonal_model, deterministic_model,
        val_loader, device
    )

    # 生成图表
    from rebuttal_lambda_max_analysis import plot_lambda_max_comparison
    plot_lambda_max_comparison(results)
    """)


if __name__ == "__main__":
    main()
