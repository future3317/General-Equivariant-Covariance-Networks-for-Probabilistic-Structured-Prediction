#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically stable loss function with eigenvalue decomposition."""

import torch
import torch.nn.functional as F
import math


def safe_eigh(A):
    """Stable eigen-decomposition with anisotropic jitter to break degeneracy."""
    device = A.device
    A = 0.5 * (A + A.transpose(-2, -1))

    dim = A.shape[-1]
    dtype = A.dtype
    base_jitter = 1e-6 if dtype == torch.float64 else 1e-4

    # Anisotropic jitter: each eigenvalue gets slightly different offset
    jitter_vec = torch.arange(dim, device=device, dtype=dtype) * base_jitter
    jitter_diag = torch.diag_embed(jitter_vec).unsqueeze(0)
    A_stable = A + jitter_diag

    try:
        return torch.linalg.eigh(A_stable)
    except Exception:
        # Fallback: add random noise
        noise_magnitude = 1e-5 if dtype == torch.float64 else 1e-3
        noise = torch.randn_like(A) * noise_magnitude
        noise_sym = 0.5 * (noise + noise.transpose(-2, -1))
        A_final = A + jitter_diag + noise_sym
        return torch.linalg.eigh(A_final)


def compute_stable_loss_paper_version(
    mu_km,
    A_km,
    targets_km,
    max_eigenvalue=3.0,  # Reduced from 5.0 to limit max variance
    min_eigenvalue=-4.0,  # Tightened from -7.0: e^-4 ≈ 0.018 min variance, prevents overconfidence
    reg_weight=1e-3,
    cond_weight=0.05,  # Re-activated: prevents condition number explosion (was 0.0, caused Cond Max > 20000)
    target_condition_number=150.0,  # Target log-condition: log(150) ≈ 5.0
    log_det_weight=1.0,  # Weight for LogDet term to control uncertainty volume (higher = more confident)
    huber_threshold=20.0,  # Huber loss threshold for D_M^2 (99.9% quantile of chi2_6 ≈ 22.5)
    use_laplacian=True,   # [NEW] Use Multivariate Laplace NLL (linear in D_M) instead of Gaussian (quadratic)
    laplacian_huber_threshold=5.0,  # [NEW] Threshold for Laplacian Huber (D_M, not D_M^2)
):
    """
    Stable negative log-likelihood loss using float64 for matrix operations.

    Inputs are in Kelvin-Mandel format (already converted by DataLoader).
    mu_km      : [B, 6] predicted mean
    A_km       : [B, 6, 6] log-covariance matrix
    targets_km : [B, 6] target values
    """
    device = mu_km.device
    orig_dtype = mu_km.dtype

    # Convert to float64 for stability
    mu_km_d = mu_km.double()
    A_km_d = A_km.double()
    targets_km_d = targets_km.double()

    # Basic validation
    if mu_km_d.shape != targets_km_d.shape:
        raise ValueError(f"mu_km shape {mu_km_d.shape} != targets_km shape {targets_km_d.shape}")
    if A_km_d.shape[-1] != 6 or A_km_d.shape[-2] != 6:
        raise ValueError(f"A_km must be [B, 6, 6], got {A_km_d.shape}")

    # Clean mu/targets BEFORE computing diff
    if not torch.isfinite(mu_km_d).all():
        print("[stable_loss] Non-finite entries in mu_km detected.")
        mu_km_d = torch.where(torch.isfinite(mu_km_d), mu_km_d, torch.zeros_like(mu_km_d))

    if not torch.isfinite(targets_km_d).all():
        print("[stable_loss] Non-finite entries in targets_km detected.")
        targets_km_d = torch.where(torch.isfinite(targets_km_d), targets_km_d, torch.zeros_like(targets_km_d))

    diff = targets_km_d - mu_km_d  # [B, 6]

    # Clean A_km (remove clamp to allow unconstrained learning)
    if not torch.isfinite(A_km_d).all():
        print("[stable_loss] Non-finite entries in A_km detected.")
        A_km_d = torch.where(torch.isfinite(A_km_d), A_km_d, torch.zeros_like(A_km_d))

    # Symmetrize
    A_km_d = 0.5 * (A_km_d + A_km_d.transpose(-2, -1))

    # Eigenvalue decomposition
    eigenvals, eigenvecs = safe_eigh(A_km_d)

    # Clamp eigenvalues
    eigenvals_clamped = torch.clamp(eigenvals, min=min_eigenvalue, max=max_eigenvalue)

    # Condition number control: log_cond <= log(200)
    max_log_cond = math.log(200.0)
    eig_min = eigenvals_clamped[:, 0]
    eig_max = eigenvals_clamped[:, -1]
    current_log_cond = eig_max - eig_min

    exceed_mask = current_log_cond > max_log_cond
    if exceed_mask.any():
        excess = (current_log_cond - max_log_cond).clamp(min=0.0)

        eig_max_adjusted = (eig_max - excess).clamp(min=eig_min)
        eigenvals_adjusted = eigenvals_clamped.clone()
        eigenvals_adjusted[exceed_mask, -1] = eig_max_adjusted[exceed_mask]
    else:
        eigenvals_adjusted = eigenvals_clamped

    # Log det: 0.5 * Tr(A) = 0.5 * sum(λ)
    # [FIX] Removed redundant * 0.5 - log_det_weight already contains the scaling
    # log_det_weight controls how strongly we penalize uncertainty volume
    log_det_term = log_det_weight * eigenvals_adjusted.sum(dim=-1).mean()

    # Mahalanobis distance computation
    # z is whitened residual: L^(-1) * (y - mu) in eigen space
    z = torch.bmm(eigenvecs.transpose(1, 2), diff.unsqueeze(-1)).squeeze(-1)

    if use_laplacian:
        # [MULTIVARIATE LAPLACE NLL]
        # Use D_M (distance) instead of D_M^2 (squared distance)
        # This is linear in the residual, matching MAE optimization
        # D_M = sqrt(sum(z_i^2 * exp(-λ_i)))
        mahalanobis_dist = torch.sqrt((z * z * torch.exp(-eigenvals_adjusted)).sum(dim=-1) + 1e-8)

        # Robust Huber-like loss for Laplacian: linear region + log tail
        # This is more stable than pure L1 for extreme outliers
        laplacian_huber_threshold_tensor = torch.tensor(laplacian_huber_threshold, device=device, dtype=mahalanobis_dist.dtype)

        # Linear region: D_M < threshold → use D_M directly
        # Tail region: D_M >= threshold → use threshold + log(1 + (D_M - threshold))
        # This prevents extreme outliers from dominating
        mahalanobis_robust = torch.where(
            mahalanobis_dist < laplacian_huber_threshold_tensor,
            mahalanobis_dist,  # Linear region (like L1)
            laplacian_huber_threshold_tensor + torch.log1p((mahalanobis_dist - laplacian_huber_threshold_tensor))
        )

        # No 0.5 factor for Laplacian (different from Gaussian)
        mahalanobis_term = mahalanobis_robust.mean()

        # For monitoring: also track raw D_M
        mahalanobis_each = mahalanobis_dist  # for compatibility with metrics
    else:
        # [ORIGINAL GAUSSIAN NLL]
        # D_M^2 for quadratic penalty (matches MSE optimization)
        mahalanobis_each = (z * z * torch.exp(-eigenvals_adjusted)).sum(dim=-1)

        # Standard Huber loss: quadratic below threshold, linear above
        huber_threshold_tensor = torch.tensor(huber_threshold, device=device, dtype=mahalanobis_each.dtype)
        quadratic_part = torch.minimum(mahalanobis_each, huber_threshold_tensor)
        linear_part = torch.clamp(mahalanobis_each - huber_threshold_tensor, min=0.0)
        huber_mahalanobis = quadratic_part + huber_threshold_tensor * linear_part.sqrt()

        # 0.5 factor for Gaussian NLL
        mahalanobis_term = 0.5 * huber_mahalanobis.mean()

    # Regularization: constrain eigenvalues
    reg_loss_high = reg_weight * torch.mean(torch.relu(eigenvals - max_eigenvalue) ** 2)
    reg_loss_low = reg_weight * torch.mean(torch.relu(min_eigenvalue - eigenvals) ** 2)

    # Condition number penalty (hinge loss)
    eig_min = eigenvals_adjusted[:, 0]
    eig_max = eigenvals_adjusted[:, -1]
    log_cond_numbers = eig_max - eig_min
    target_log_cond = math.log(target_condition_number)
    hinge_penalty = torch.relu(log_cond_numbers - target_log_cond)
    reg_condition = cond_weight * torch.mean(hinge_penalty)

    # Debug: print reg_condition details
    if reg_condition > 0:
        print(f"[DEBUG] cond_weight={cond_weight}, exceed_ratio={(hinge_penalty>0).float().mean().item():.3f}, "
              f"mean_hinge={hinge_penalty.mean().item():.4f}, reg_condition={reg_condition.item():.6f}")

    loss = log_det_term + mahalanobis_term + reg_loss_high + reg_loss_low + reg_condition

    # Monitoring metrics (detached)
    cond_numbers = torch.exp(log_cond_numbers)

    # Additional metrics for Laplacian mode
    if use_laplacian:
        # For Laplacian: track D_M (distance) instead of D_M^2
        components = {
            "loss_fit": mahalanobis_term.detach(),
            "loss_uncertainty": log_det_term.detach(),
            "loss_reg": (reg_loss_high + reg_loss_low + reg_condition).detach(),
            "loss_reg_high": reg_loss_high.detach(),
            "loss_reg_low": reg_loss_low.detach(),
            "loss_reg_cond": reg_condition.detach(),
            "cond_number_mean": torch.mean(cond_numbers).detach(),
            "cond_number_max": torch.max(cond_numbers).detach(),
            "cond_exceed_ratio": torch.mean((cond_numbers > target_condition_number).float()).detach(),
            # Laplacian-specific metrics
            "mahalanobis_dist_mean": mahalanobis_dist.mean().detach(),
            "mahalanobis_dist_median": mahalanobis_dist.median().detach(),
            "mahalanobis_dist_max": mahalanobis_dist.max().detach(),
            "mahalanobis_exceed_ratio": torch.mean((mahalanobis_dist > laplacian_huber_threshold).float()).detach(),
        }
    else:
        # Gaussian metrics
        components = {
            "loss_fit": mahalanobis_term.detach(),
            "loss_uncertainty": log_det_term.detach(),
            "loss_reg": (reg_loss_high + reg_loss_low + reg_condition).detach(),
            "loss_reg_high": reg_loss_high.detach(),
            "loss_reg_low": reg_loss_low.detach(),
            "loss_reg_cond": reg_condition.detach(),
            "cond_number_mean": torch.mean(cond_numbers).detach(),
            "cond_number_max": torch.max(cond_numbers).detach(),
            "cond_exceed_ratio": torch.mean((cond_numbers > target_condition_number).float()).detach(),
        }

    # Fallback to MSE if loss is NaN
    if not torch.isfinite(loss):
        print(f"Warning: Non-finite loss detected!")
        print(f"  eigenvals range: [{eigenvals.min().item():.3f}, {eigenvals.max().item():.3f}]")
        print(f"  log_det_term: {log_det_term.item():.3f}")
        print(f"  mahalanobis_term: {mahalanobis_term.item():.3f}")

        fallback_loss = torch.nn.functional.mse_loss(mu_km_d, targets_km_d)
        print(f"  -> Using MSE fallback: {fallback_loss.item():.6f}")
        loss = fallback_loss

        zero_tensor = torch.tensor(0.0, device=device, dtype=mu_km_d.dtype)
        components = {
            "loss_fit": fallback_loss.detach(),
            "loss_uncertainty": zero_tensor.clone(),
            "loss_reg": zero_tensor.clone(),
            "loss_reg_high": zero_tensor.clone(),
            "loss_reg_low": zero_tensor.clone(),
            "loss_reg_cond": zero_tensor.clone(),
            "cond_number_mean": zero_tensor.clone(),
            "cond_number_max": zero_tensor.clone(),
            "cond_exceed_ratio": zero_tensor.clone(),
        }

    # Convert back to original dtype
    loss_final = loss.to(dtype=orig_dtype)
    for key in components:
        components[key] = components[key].to(dtype=orig_dtype)

    return loss_final, components


def test_stable_loss():
    """测试稳定损失函数（Gaussian 和 Laplacian 模式）"""
    print("="*60)
    print("测试多元拉普拉斯损失 vs 高斯损失")
    print("="*60)

    # 创建测试数据
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试两种输入精度
    for dtype_name, dtype in [("float32", torch.float32), ("float64", torch.float64)]:
        print(f"\n--- 测试输入精度: {dtype_name} ---")

        # 预测值
        mu_voigt = torch.randn(batch_size, 6, device=device, dtype=dtype) * 0.1
        targets_voigt = torch.randn(batch_size, 6, device=device, dtype=dtype) * 0.1

        # 创建协方差矩阵A (在Kelvin-Mandel空间)
        A_km = torch.randn(batch_size, 6, 6, device=device, dtype=dtype) * 0.1
        A_km = 0.5 * (A_km + A_km.transpose(-2, -1))  # 确保对称

        print(f"输入数据范围:")
        print(f"  mu_voigt: [{mu_voigt.min():.3f}, {mu_voigt.max():.3f}]")
        print(f"  targets_voigt: [{targets_voigt.min():.3f}, {targets_voigt.max():.3f}]")
        print(f"  A_km: [{A_km.min():.3f}, {A_km.max():.3f}]")

        # 测试 Gaussian NLL（原始版本）
        print(f"\n  [Gaussian NLL]")
        mu_voigt_g = mu_voigt.clone().detach()
        A_km_g = A_km.clone().detach()
        mu_voigt_g.requires_grad_(True)
        A_km_g.requires_grad_(True)

        loss_g, comp_g = compute_stable_loss_paper_version(
            mu_voigt_g, A_km_g, targets_voigt,
            use_laplacian=False
        )
        print(f"    loss: {loss_g.item():.6f}")
        print(f"    loss_fit: {comp_g['loss_fit'].item():.4f}")
        print(f"    loss_uncertainty: {comp_g['loss_uncertainty'].item():.4f}")

        loss_g.backward()
        grad_norm_mu_g = mu_voigt_g.grad.norm().item() if mu_voigt_g.grad is not None else float('nan')
        print(f"    mu grad norm: {grad_norm_mu_g:.6f}")

        # 测试 Laplacian NLL（新版本）
        print(f"\n  [Laplacian NLL]")
        mu_voigt_l = mu_voigt.clone().detach()
        A_km_l = A_km.clone().detach()
        mu_voigt_l.requires_grad_(True)
        A_km_l.requires_grad_(True)

        loss_l, comp_l = compute_stable_loss_paper_version(
            mu_voigt_l, A_km_l, targets_voigt,
            use_laplacian=True,
            laplacian_huber_threshold=5.0
        )
        print(f"    loss: {loss_l.item():.6f}")
        print(f"    loss_fit: {comp_l['loss_fit'].item():.4f}")
        print(f"    loss_uncertainty: {comp_l['loss_uncertainty'].item():.4f}")
        if 'mahalanobis_dist_mean' in comp_l:
            print(f"    D_M mean: {comp_l['mahalanobis_dist_mean'].item():.4f}")
            print(f"    D_M median: {comp_l['mahalanobis_dist_median'].item():.4f}")
            print(f"    D_M max: {comp_l['mahalanobis_dist_max'].item():.4f}")

        loss_l.backward()
        grad_norm_mu_l = mu_voigt_l.grad.norm().item() if mu_voigt_l.grad is not None else float('nan')
        print(f"    mu grad norm: {grad_norm_mu_l:.6f}")

        # 比较
        print(f"\n  [对比]")
        print(f"    Laplacian/Gaussian loss ratio: {loss_l.item() / loss_g.item():.3f}")
        print(f"    Laplacian/Gaussian grad ratio: {grad_norm_mu_l / grad_norm_mu_g:.3f}")

        # 检查是否有NaN梯度
        has_nan_grad = False
        for name, param in [('mu_lap', mu_voigt_l), ('A_lap', A_km_l)]:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"    [FAIL] NaN gradient in {name}!")
                    has_nan_grad = True

        if not has_nan_grad:
            print(f"    [PASS] {dtype_name} 所有检查通过！")

    print("\n" + "="*60)
    print("多元拉普拉斯损失的优势：")
    print("1. D_M 线性增长，与 MAE 目标一致（统计学上对应拉普拉斯分布）")
    print("2. 对离群点更鲁棒（材料数据常有长尾分布）")
    print("3. 梯度更稳定，不会因大误差而爆炸")
    print("4. 保留协方差结构，捕捉分量间相关性")


if __name__ == "__main__":
    test_stable_loss()