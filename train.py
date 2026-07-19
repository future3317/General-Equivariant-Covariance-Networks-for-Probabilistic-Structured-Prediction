"""
train.py
--------
Training script for E(3)-equivariant uncertainty quantification model.
Now configured for dielectric tensor prediction with optimized packed data.
All preprocessing (edges, Voigt, Log, normalization) is done in preprocess_edges.py.
"""
import os
# Note: expandable_segments not supported on this platform, removed to avoid warnings
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import logging
from datetime import datetime
import torchmetrics
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, MeanMetric

from dielectric_data_loader import get_dielectric_data_loaders
from equivariant_network import EquivariantUncertaintyNetwork
from voigt_utils import voigt_to_kelvin_mandel, kelvin_mandel_to_voigt, sym_matrix_exp_voigt
from stable_loss_implementation import compute_stable_loss_paper_version


class SpdRate(torchmetrics.Metric):
    """Custom metric to calculate the rate of positive definite matrices."""
    def __init__(self):
        super().__init__()
        self.add_state("pos_def_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, matrix_pred):
        # matrix_pred: [B, 3, 3] in physical space
        eigs = torch.linalg.eigvalsh(matrix_pred)
        is_pos_def = (eigs[:, 0] > 0)  # Smallest eigenvalue > 0
        self.pos_def_count += is_pos_def.sum()
        self.total += matrix_pred.shape[0]

    def compute(self):
        return self.pos_def_count.float() / self.total


def setup_logger(save_dir, experiment_name=None):
    """Setup logger for training session."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"training_{timestamp}"

    log_file = os.path.join(save_dir, f"{experiment_name}.log")
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, experiment_name


def log_epoch_metrics(logger, epoch, train_loss, val_metrics, learning_rate, train_loss_components=None):
    """Log detailed metrics for each epoch."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Epoch {epoch + 1} Summary")
    logger.info(f"{'='*60}")

    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
    logger.info(f"Val MAE: {val_metrics['mae']:.4f} (Total)")

    # Log training loss components if available
    if train_loss_components:
        logger.info(f"[Train Loss Decomposition]")
        logger.info(f"  Data Fit: {train_loss_components['loss_fit']:.4f}")
        logger.info(f"  Uncertainty: {train_loss_components['loss_uncertainty']:.4f}")
        logger.info(f"  Regularization: {train_loss_components['loss_reg']:.4f}")

    if 'mae_diag' in val_metrics:
        logger.info(f"Val MAE Diag: {val_metrics['mae_diag']:.4f}")
        logger.info(f"Val MAE Off: {val_metrics['mae_off_diag']:.4f}")

    # [FIX] Removed RMSE logging - not useful due to outliers
    logger.info(f"Val MAPE: {val_metrics['mape']:.2%}")  # Show as percentage

    # New: Additional evaluation metrics
    if 'mae_median' in val_metrics:
        logger.info(f"Val MAE Median: {val_metrics['mae_median']:.4f}")
    if 'log_mae' in val_metrics:
        logger.info(f"Val Log-MAE: {val_metrics['log_mae']:.4f}")

    # Compare Log-MAE vs Physical MAE for insight
    if 'log_mae' in val_metrics and val_metrics['mae'] > 0:
        ratio = val_metrics['log_mae'] / val_metrics['mae']
        if ratio < 0.5:
            logger.info(f"[Insight] Log-MAE << Physical-MAE (ratio={ratio:.2f}) - Model learns relative scaling well!")
        elif ratio > 2.0:
            logger.info(f"[Insight] Log-MAE >> Physical-MAE (ratio={ratio:.2f}) - Issues with large-scale predictions")

    logger.info(f"[Uncertainty Health]")
    if 'trace_sigma' in val_metrics:
        logger.info(f"Tr(Sigma): {val_metrics['trace_sigma']:.3f}")
    if 'log_det_sigma' in val_metrics:
        logger.info(f"LogDet(Sigma): {val_metrics['log_det_sigma']:.3f}")
    if 'cond_sigma_mean' in val_metrics:
        logger.info(f"Cond(Sigma): {val_metrics['cond_sigma_mean']:.2f} (Max: {val_metrics['cond_sigma_max']:.1f})")
    if 'cond_number_mean' in val_metrics:
        logger.info(f"Cond Number Avg: {val_metrics['cond_number_mean']:.1f} (Exceed Ratio: {val_metrics['cond_exceed_ratio']:.1%})")
    if 'mahalanobis' in val_metrics:
        logger.info(f"Mahalanobis RMS: {val_metrics['mahalanobis']:.3f} (Median: {val_metrics['mahalanobis_median']:.3f})")

    logger.info(f"[Physics]")
    if 'pos_def_rate' in val_metrics:
        logger.info(f"Pos. Definite: {val_metrics['pos_def_rate']:.1f}%")
    if 'anisotropy_mae' in val_metrics:
        logger.info(f"Anisotropy MAE: {val_metrics['anisotropy_mae']:.4f}")
    # [FIX] Removed eigenvalue_mae - redundant with mae_diag

    logger.info(f"[Loss Decomposition]")
    if 'loss_fit' in val_metrics:
        logger.info(f"Data Fit (Mahalanobis): {val_metrics['loss_fit']:.4f}")
        logger.info(f"Uncertainty (LogDet): {val_metrics['loss_uncertainty']:.4f}")
        logger.info(f"Regularization: {val_metrics['loss_reg']:.4f}")

    logger.info(f"Learning Rate: {learning_rate:.6f}")

    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'val_loss': float(val_metrics['loss']),
        'val_mae': float(val_metrics['mae']),
        # [FIX] Removed val_rmse - not useful due to outliers
        'learning_rate': float(learning_rate)
    }

    # Add new metrics if available
    if 'mae_median' in val_metrics:
        epoch_metrics['val_mae_median'] = float(val_metrics['mae_median'])
    if 'log_mae' in val_metrics:
        epoch_metrics['val_log_mae'] = float(val_metrics['log_mae'])
    if 'mae_diag' in val_metrics:
        epoch_metrics['val_mae_diag'] = float(val_metrics['mae_diag'])
        epoch_metrics['val_mae_off_diag'] = float(val_metrics['mae_off_diag'])
    if 'trace_sigma' in val_metrics:
        epoch_metrics['trace_sigma'] = float(val_metrics['trace_sigma'])
    if 'log_det_sigma' in val_metrics:
        epoch_metrics['log_det_sigma'] = float(val_metrics['log_det_sigma'])
    if 'cond_sigma_mean' in val_metrics:
        epoch_metrics['cond_sigma_mean'] = float(val_metrics['cond_sigma_mean'])
        epoch_metrics['cond_sigma_max'] = float(val_metrics['cond_sigma_max'])
    if 'mahalanobis' in val_metrics:
        epoch_metrics['mahalanobis'] = float(val_metrics['mahalanobis'])
    if 'pos_def_rate' in val_metrics:
        epoch_metrics['pos_def_rate'] = float(val_metrics['pos_def_rate'])
    if 'anisotropy_mae' in val_metrics:
        epoch_metrics['anisotropy_mae'] = float(val_metrics['anisotropy_mae'])
    if 'mape' in val_metrics:
        epoch_metrics['mape'] = float(val_metrics['mape'])
    # [FIX] Removed eigenvalue_mae - redundant with mae_diag
    if 'loss_fit' in val_metrics:
        epoch_metrics['val_loss_fit'] = float(val_metrics['loss_fit'])
        epoch_metrics['val_loss_uncertainty'] = float(val_metrics['loss_uncertainty'])
        epoch_metrics['val_loss_reg'] = float(val_metrics['loss_reg'])

    return epoch_metrics




def train_epoch(model, dataloader, optimizer, device, epoch=0, scheduler=None):
    """Train for one epoch with optimized GPU-CPU synchronization."""
    model.train()

    # Initialize accumulators as tensors on device to avoid CPU-GPU sync
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0, device=device)

    # Track loss components as tensors on GPU
    loss_fit_sum = torch.tensor(0.0, device=device)
    loss_uncertainty_sum = torch.tensor(0.0, device=device)
    loss_reg_sum = torch.tensor(0.0, device=device)

    pbar = tqdm(dataloader, desc="Training",
                bar_format='{l_bar}{bar}| {elapsed} {rate_fmt}{postfix}',
                mininterval=1.0,
                postfix={'bs': f'{dataloader.batch_size}'})

    for batch_idx_iter, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        batch = batch.to(device)

        # Skip if no edges (graph is empty)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            print(f"Warning: Batch {batch_idx_iter} has no edges, skipping...")
            continue

        # Use PyG Data object directly (model expects Data object, not dict)
        # No need to convert to dictionary

        mu_km, A_km, _ = model(batch, compute_sigma=False)

        # 优化训练策略：detach() + 高 w_mse 的科学组合
        #
        # detach() 的作用：
        # 1. 切断 UQ 分支对 feature_network 的梯度回流
        # 2. feature_network 成为纯粹的物理特征提取器，只听 MSE
        # 3. UQ 分支被迫学习"如何从固定特征中识别误差"
        #
        # 高 w_mse 的必要性：
        # 1. 确保骨架学到有意义的物理特征
        # 2. 防止特征退化导致 UQ 无从评估
        # 3. 避免"Shortcut to Ignorance"

        # [TRAINING STRATEGY] Laplacian NLL is MAE-friendly
        # - Mahalanobis term now uses D_M (linear) instead of D_M^2 (quadratic)
        # - Less need for MSE warmup since NLL already aligns with MAE goal
        # - Keep small MSE for numerical stability in early epochs

        if epoch < 5:
            # Brief warmup: small MSE for stability
            w_mse = 0.1
            w_nll = 1.0
        else:
            # Laplacian NLL主导: already MAE-friendly
            w_mse = 0.01   # Keep tiny MSE for gradient stability
            w_nll = 1.0


        # Reshape batch.y from [num_nodes * 6] to [batch_size, 6]
        targets = batch.y.view(mu_km.shape[0], 6)

        # [LAPLACIAN NLL] Multivariate Laplace NLL for MAE-friendly training
        # - Linear in D_M (distance) instead of D_M^2 (squared distance)
        # - Matches MAE optimization goal statistically
        # - More robust to outliers in material property data

        # [FIX] Use model's eigenvalue bounds instead of local constants
        # This ensures consistency between forward pass (model) and loss function
        nll_loss, nll_components = compute_stable_loss_paper_version(
            mu_km, A_km, targets,
            use_laplacian=True,              # [NEW] Enable Multivariate Laplace NLL
            laplacian_huber_threshold=5.0,   # [NEW] D_M threshold for Huber (not squared!)
            cond_weight=0.5,                 # Increased: prevents cond explosion
            log_det_weight=0.1,              # Controls uncertainty volume (reduced from 0.5 to allow larger uncertainty)
            min_eigenvalue=model.min_log_eigenvalue,  # [FIX] Read from model
            max_eigenvalue=model.max_log_eigenvalue,  # [FIX] Read from model
            reg_weight=1e-3,
            target_condition_number=100.0,   # Tighter condition number target
        )

        mse_loss = torch.nn.functional.mse_loss(mu_km, targets)

        # Track loss components - accumulate as tensors on GPU
        if 'loss_fit' in nll_components:
            loss_fit_sum += nll_components['loss_fit'].detach()
            loss_uncertainty_sum += nll_components['loss_uncertainty'].detach()
            loss_reg_sum += nll_components['loss_reg'].detach()

        # 计算总 Loss
        loss = w_mse * mse_loss + w_nll * nll_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n  [CRITICAL] Non-finite gradients detected! grad_norm={grad_norm:.3f}")
            print(f"  Locating first parameter with NaN/Inf gradients...")

            # 定位第一个有问题的参数
            first_invalid_param = None
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        first_invalid_param = (name, param)
                        break

            if first_invalid_param:
                name, param = first_invalid_param
                grad = param.grad
                print(f"  First invalid parameter: '{name}'")
                print(f"    Parameter shape: {param.shape}")
                print(f"    Parameter device: {param.device}")
                print(f"    Parameter dtype: {param.dtype}")
                print(f"    Gradient statistics:")
                print(f"      min: {grad.min().item():.6e}")
                print(f"      max: {grad.max().item():.6e}")
                print(f"      mean: {grad.mean().item():.6e}")
                print(f"      norm: {grad.norm().item():.6e}")

                # 检查 NaN/Inf 分布
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                print(f"      NaN elements: {nan_count}/{grad.numel()}")
                print(f"      Inf elements: {inf_count}/{grad.numel()}")

                # 如果是 cov_mlp 的参数，给出特别提示
                if 'cov_mlp' in name:
                    print(f"    >>> This is a cov_mlp parameter! Consider checking NLL computation <<<")
                elif 'mean_head' in name:
                    print(f"    >>> This is a mean_head parameter <<<")
                elif 'feature_network' in name or 'equivariant' in name:
                    print(f"    >>> This is a feature network parameter <<<")
            else:
                print(f"  No parameters with non-finite gradients found (unexpected!)")

            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.detach()
        num_batches += 1

        # [简化] 只在进度条显示 loss 和 mse
        if batch_idx_iter % 10 == 0:  # 限制更新频率以减少开销
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'mse': f"{mse_loss.item():.3f}"
            })

    # Final synchronization only once at the end
    num_batches_cpu = num_batches.item()

    # Return detailed loss information - sync only once at the end
    if num_batches_cpu > 0:
        avg_loss = (total_loss / num_batches).item()
        avg_loss_fit = (loss_fit_sum / num_batches).item()
        avg_loss_uncertainty = (loss_uncertainty_sum / num_batches).item()
        avg_loss_reg = (loss_reg_sum / num_batches).item()
    else:
        avg_loss = float('nan')
        avg_loss_fit = avg_loss_uncertainty = avg_loss_reg = 0.0

    return avg_loss, {
        'loss_fit': avg_loss_fit,
        'loss_uncertainty': avg_loss_uncertainty,
        'loss_reg': avg_loss_reg
    }


def validate(model, dataloader, device, metrics_collection, scaler=None, train_log_mean=None, train_log_std=None):
    """Validate the model using torchmetrics for cleaner and safer metrics computation.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        device: Device to run on
        metrics_collection: Pre-created MetricCollection object (reused across epochs)
        scaler: Unused scaler (for compatibility)
        train_log_mean: Training set log mean (must use training set statistics!)
        train_log_std: Training set log std (must use training set statistics!)
    """
    model.eval()

    # [修复] 重置指标状态，防止内存累积
    metrics_collection.reset()
    # metrics_collection 已在创建时设置为 CPU，无需每次 to('cpu')

    # Use training set normalization parameters (CRITICAL!)
    # [FIX] Component-wise normalization for proper denormalization
    LOG_MEAN_SCALAR = train_log_mean if train_log_mean is not None else dataloader.dataset.log_mean
    LOG_STD_SCALAR = train_log_std if train_log_std is not None else dataloader.dataset.log_std
    COMPONENT_MEAN = dataloader.dataset.component_mean  # [6]
    COMPONENT_STD = dataloader.dataset.component_std      # [6]

    loss_avg = torch.tensor(0.0, device='cpu')

    # For loss decomposition - use CPU to save GPU memory
    loss_fit_avg = torch.tensor(0.0, device='cpu')
    loss_uncertainty_avg = torch.tensor(0.0, device='cpu')
    loss_reg_avg = torch.tensor(0.0, device='cpu')

    # For uncertainty metrics - use CPU to save GPU memory
    trace_sigma_avg = torch.tensor(0.0, device='cpu')
    log_det_sigma_avg = torch.tensor(0.0, device='cpu')
    cond_sigma_avg = torch.tensor(0.0, device='cpu')
    cond_sigma_max_tracker = []  # Track maximum condition number
    mahalanobis_avg = torch.tensor(0.0, device='cpu')
    # [FIX] Removed eigenvalue_mae_avg - redundant with mae_diag
    norm_A_avg = torch.tensor(0.0, device='cpu')
    mape_avg = torch.tensor(0.0, device='cpu')  # New: Mean Absolute Percentage Error

    # Pre-allocate fixed-size buffers to avoid dynamic list growth and reduce CPU/numpy conversions
    # Estimate maximum number of samples (adjust based on your dataset size)
    # Note: each sample has 6 values for MAE, so we need 6x space
    dataset = dataloader.dataset
    if hasattr(dataset, '__len__'):
        try:
            dataset_size = len(dataset)
            max_samples = min(10000, dataset_size)
        except:
            max_samples = 10000
    else:
        max_samples = 10000
    mae_median_buffer = torch.zeros(max_samples * 6, device='cpu')  # 6 values per sample
    mahalanobis_median_buffer = torch.zeros(max_samples, device='cpu')
    mae_idx = 0  # Separate index for MAE buffer
    mah_idx = 0  # Separate index for Mahalanobis buffer

    # Additional accumulators
    batch_count = 0
    log_mae_sum = 0.0  # Sum of log-space MAE
    log_mae_count = 0  # Count for log-space MAE

    # Pre-create mean_vec tensor once (outside the loop)
    mean_vec = torch.tensor([LOG_MEAN_SCALAR]*3 + [0.0]*3, device=device).unsqueeze(0)

    with torch.inference_mode():
        for batch_idx_iter, batch in enumerate(tqdm(dataloader, desc="Validation",
                         bar_format='{l_bar}{bar}| {elapsed} {rate_fmt}',
                         postfix={'bs': f'{dataloader.batch_size}'})):
            # Move batch to device
            batch = batch.to(device) if hasattr(batch, 'to') else batch

            # Skip if no edges
            if not hasattr(batch, 'edge_index') or batch.edge_index is None or batch.edge_index.numel() == 0:
                print("Warning: Batch has no edges, skipping...")
                continue

            # Forward pass
            mu_km, A_km, Sigma_km = model(batch, compute_sigma=True)

            # Reshape batch.y for proper batch dimensions
            targets = batch.y.view(mu_km.shape[0], 6)


            # Compute loss (use same Laplacian parameters as training)
            # [FIX] Use model's eigenvalue bounds instead of hardcoded values
            batch_loss, loss_components = compute_stable_loss_paper_version(
                mu_km, A_km, targets,
                use_laplacian=True,
                laplacian_huber_threshold=5.0,
                cond_weight=0.5,
                log_det_weight=0.1,  # Reduced from 0.5 to allow larger uncertainty
                min_eigenvalue=model.min_log_eigenvalue,  # [FIX] Read from model
                max_eigenvalue=model.max_log_eigenvalue,  # [FIX] Read from model
                reg_weight=1e-3,
                target_condition_number=100.0,
            )
            loss_avg += batch_loss.item()
            batch_count += 1

            # Update loss components
            if 'loss_fit' in loss_components:
                loss_fit_avg += loss_components['loss_fit'].cpu().item()
                loss_uncertainty_avg += loss_components['loss_uncertainty'].cpu().item()
                loss_reg_avg += loss_components['loss_reg'].cpu().item()

            # Convert to physical space
            # Note: mu_km and targets are in Kelvin-Mandel space from the data loader
            # Need to convert back to Standard Voigt for matrix operations
            mu_voigt_std = kelvin_mandel_to_voigt(mu_km)
            targets_std = kelvin_mandel_to_voigt(targets)

            # [FIX] Component-wise denormalization
            mean_vec = torch.tensor(COMPONENT_MEAN, device=mu_voigt_std.device, dtype=mu_voigt_std.dtype)
            std_vec = torch.tensor(COMPONENT_STD, device=mu_voigt_std.device, dtype=mu_voigt_std.dtype)
            mu_log = mu_voigt_std * std_vec + mean_vec
            target_log = targets_std * std_vec + mean_vec

            # [OPTIMIZATION] Use optimized function that returns both Voigt and Matrix formats
            # This avoids redundant Voigt -> Matrix -> Exp -> Voigt -> Matrix conversions
            mu_phys, mu_phys_matrix = sym_matrix_exp_voigt_with_matrix(mu_log)
            target_phys, target_phys_matrix = sym_matrix_exp_voigt_with_matrix(target_log)

            # Move to CPU for metrics to avoid device mismatch and reduce memory usage
            # detach() is redundant inside torch.inference_mode()
            mu_cpu = mu_phys.cpu()
            target_cpu = target_phys.cpu()

            # [OPTIMIZATION] Directly use matrix form for SpdRate (no redundant conversion)
            pred_matrices_cpu = mu_phys_matrix.cpu()
            target_matrices_cpu = target_phys_matrix.cpu()
            metrics_collection['spd_rate'].update(pred_matrices_cpu)

            # Update main metrics (excluding spd_rate)
            metrics_collection['mae'].update(mu_cpu, target_cpu)
            # [FIX] Removed rmse.update() - not useful due to outliers

            # Update diagonal/off-diagonal MAE separately
            # Must use .contiguous() after slicing to make tensors memory-contiguous
            metrics_collection['mae_diag'].update(
                mu_cpu[:, :3].contiguous(),
                target_cpu[:, :3].contiguous()
            )
            metrics_collection['mae_off_diag'].update(
                mu_cpu[:, 3:].contiguous(),
                target_cpu[:, 3:].contiguous()
            )

            # Calculate additional metrics using fixed buffers (use CPU tensors)
            # 1. Collect absolute errors for median MAE (avoid numpy conversion)
            abs_errors = torch.abs(mu_cpu - target_cpu)
            num_errors = abs_errors.numel()

            # Store in fixed buffer to avoid dynamic list growth
            if mae_idx + num_errors <= len(mae_median_buffer):
                mae_median_buffer[mae_idx:mae_idx+num_errors] = abs_errors.flatten()
                mae_idx += num_errors
            else:
                # Buffer full - compute median from current data and continue
                if mae_idx > 0:
                    current_median = torch.median(mae_median_buffer[:mae_idx])
                    print(f"MAE buffer full, median so far: {current_median.item():.4f}")
                mae_idx = 0  # Reset for next epoch if needed

            # 2. Calculate Log-MAE (for non-zero values)
            # Use a small epsilon to avoid log(0)
            epsilon = 1e-6
            non_zero_mask = (torch.abs(mu_cpu) > epsilon) | (torch.abs(target_cpu) > epsilon)
            if non_zero_mask.any():
                log_mu = torch.log(torch.abs(mu_cpu[non_zero_mask]) + epsilon)
                log_target = torch.log(torch.abs(target_cpu[non_zero_mask]) + epsilon)
                log_mae_sum += torch.abs(log_mu - log_target).sum().item()
                log_mae_count += non_zero_mask.sum().item()

            # Calculate MAPE (Mean Absolute Percentage Error) - only for diagonal elements (xx, yy, zz)
            # This avoids numerical explosion from near-zero off-diagonal values - use CPU tensors
            diag_errors = torch.abs(mu_cpu[:, :3] - target_cpu[:, :3])
            diag_targets = torch.abs(target_cpu[:, :3])
            # Only calculate MAPE for values > 0.1 to avoid division by very small numbers
            mask = diag_targets > 0.1
            if mask.sum() > 0:
                mape_score = (diag_errors[mask] / diag_targets[mask]).mean()
                mape_avg += mape_score.item()

            # Calculate Anisotropy Error
            # [FIX] Use pre-computed matrices instead of redundant conversion
            # pred_matrices_cpu and target_matrices_cpu are already computed above

            # Get eigenvalues for anisotropy calculation
            pred_evals = torch.linalg.eigvalsh(pred_matrices_cpu)  # [B, 3]
            true_evals = torch.linalg.eigvalsh(target_matrices_cpu)  # [B, 3]

            # Calculate anisotropy (max - min eigenvalue)
            pred_anisotropy = pred_evals[:, -1] - pred_evals[:, 0]  # max - min
            true_anisotropy = true_evals[:, -1] - true_evals[:, 0]

            # Update anisotropy MAE
            metrics_collection['anisotropy_mae'].update(pred_anisotropy, true_anisotropy)

            # Update uncertainty metrics if available
            if Sigma_km is not None:
                sigma_eigs = torch.linalg.eigvalsh(Sigma_km)
                trace_sigma_avg += torch.diagonal(Sigma_km, dim1=-2, dim2=-1).sum(dim=-1).mean().item()
                log_det_sigma_avg += torch.sum(torch.log(sigma_eigs + 1e-6), dim=-1).mean().item()

                # Condition number (moved inside the if block)
                cond = sigma_eigs[:, -1] / (sigma_eigs[:, 0] + 1e-6)
                cond_sigma_avg += cond.mean().item()
                cond_sigma_max_tracker.append(cond.max().item())

                # Mahalanobis distance computation (fixed - no duplicate conversion)
                try:
                    # 1. targets and mu_km are already in Kelvin-Mandel space
                    # targets has been reshaped to [B, 6] earlier, mu_km is [B, 6]
                    targets_km = targets  # Already in KM space, already reshaped
                    # mu_km is already in KM space from model forward pass

                    # 2. Calculate difference [B, 6, 1]
                    diff_km = (targets_km - mu_km).unsqueeze(-1)

                    # 3. Solve Sigma * x = diff (more stable than inversion)
                    inv_sigma_diff = torch.linalg.solve(Sigma_km, diff_km)

                    # 4. Calculate distance: sqrt(diff^T * Sigma^-1 * diff)
                    mahalanobis_sq = torch.bmm(diff_km.transpose(1, 2), inv_sigma_diff).squeeze()

                    # 5. Update with safe computation
                    # Compute true RMS: sqrt(mean of squared distances)
                    mahalanobis_rms = torch.sqrt(mahalanobis_sq.mean())
                    mahalanobis_avg += mahalanobis_rms.item()

                    # Store Mahalanobis distances in fixed buffer (avoid numpy conversion)
                    mahalanobis_values = mahalanobis_sq.sqrt()
                    num_values = len(mahalanobis_values)

                    if mah_idx + num_values <= len(mahalanobis_median_buffer):
                        mahalanobis_median_buffer[mah_idx:mah_idx+num_values] = mahalanobis_values
                        mah_idx += num_values
                    else:
                        # Buffer full - skip remaining to avoid overflow
                        print("Mahalanobis buffer full, skipping remaining distances")
                except Exception as e:
                    # Print error only for first batch to avoid spamming logs
                    if batch_idx_iter == 0:
                        print(f"Mahalanobis calc error: {e}")
                    pass

            # Update other metrics
            norm_A_avg += torch.norm(A_km, p='fro', dim=(1, 2)).mean().item()

            # [FIX] Removed eigenvalue_mae computation - redundant with mae_diag

    # Compute all metrics safely
    computed_metrics = metrics_collection.compute()
    final_loss = loss_avg / max(batch_count, 1)

    # Build results dictionary
    # [FIX] Removed 'rmse' and 'eigenvalue_mae' - not useful for this task
    results = {
        'loss': final_loss.item(),
        'mae': computed_metrics['mae'].item(),
        'mse': (computed_metrics['mae'].item()) ** 2,  # Approximate MSE from MAE
        'mae_diag': computed_metrics['mae_diag'].item(),
        'mae_off_diag': computed_metrics['mae_off_diag'].item(),
        'pos_def_rate': computed_metrics['spd_rate'].item() * 100.0,
        'anisotropy_mae': computed_metrics['anisotropy_mae'].item(),
    }

    # Add loss components
    results['loss_fit'] = loss_fit_avg / max(batch_count, 1)
    results['loss_uncertainty'] = loss_uncertainty_avg / max(batch_count, 1)
    results['loss_reg'] = loss_reg_avg / max(batch_count, 1)

    # Add uncertainty metrics with safe computation
    # Use a helper function to safely compute metrics
    # Calculate medians from tensor buffers (avoid numpy conversion)
    mae_median = torch.median(mae_median_buffer[:mae_idx]).item() if mae_idx > 0 else 0.0
    mahalanobis_median = torch.median(mahalanobis_median_buffer[:mah_idx]).item() if mah_idx > 0 else 0.0

    results.update({
        'trace_sigma': trace_sigma_avg / max(batch_count, 1),
        'log_det_sigma': log_det_sigma_avg / max(batch_count, 1),
        'cond_sigma_mean': cond_sigma_avg / max(batch_count, 1),
        'cond_sigma_max': max(cond_sigma_max_tracker) if cond_sigma_max_tracker else 0.0,
        'mahalanobis': mahalanobis_avg / max(batch_count, 1),
        'mahalanobis_median': mahalanobis_median,
        'norm_A': norm_A_avg / max(batch_count, 1),
        # [FIX] Removed eigenvalue_mae - redundant with mae_diag
        'mape': mape_avg / max(batch_count, 1),
        # Additional evaluation metrics
        'mae_median': mae_median,
        'log_mae': (log_mae_sum / log_mae_count) if log_mae_count > 0 else 0.0,
    })

    return results


def voigt_to_matrix_batch(voigt_vectors):
    """Convert batch of Voigt notation vectors to 3x3 matrices."""
    batch_size = voigt_vectors.shape[0]
    matrices = torch.zeros(batch_size, 3, 3, device=voigt_vectors.device)
    matrices[:, 0, 0] = voigt_vectors[:, 0]
    matrices[:, 1, 1] = voigt_vectors[:, 1]
    matrices[:, 2, 2] = voigt_vectors[:, 2]
    matrices[:, 1, 2] = voigt_vectors[:, 3]
    matrices[:, 2, 1] = voigt_vectors[:, 3]
    matrices[:, 0, 2] = voigt_vectors[:, 4]
    matrices[:, 2, 0] = voigt_vectors[:, 4]
    matrices[:, 0, 1] = voigt_vectors[:, 5]
    matrices[:, 1, 0] = voigt_vectors[:, 5]
    return matrices


def sym_matrix_exp_voigt_with_matrix(voigt_tensor):
    """
    [OPTIMIZATION] 计算矩阵指数并同时返回 Voigt 和 矩阵格式。

    避免冗余转换: Voigt -> Matrix -> Exp -> Voigt -> Matrix

    Args:
        voigt_tensor: [B, 6] Voigt representation (log space)

    Returns:
        exp_voigt: [B, 6] Voigt representation (for MAE metrics)
        exp_matrix: [B, 3, 3] Matrix form (for SpdRate and eigenvalue checks)
    """
    matrices = voigt_to_matrix_batch(voigt_tensor)
    orig_dtype = matrices.dtype
    device = matrices.device

    # 使用双精度进行特征值分解
    matrices_d = matrices.double()
    # 强制对称化以消除数值噪声
    matrices_d = 0.5 * (matrices_d + matrices_d.transpose(-2, -1))

    L, Q = torch.linalg.eigh(matrices_d)

    # 指数转换 + 溢出保护
    L_exp = torch.exp(torch.clamp(L, max=20.0))

    # 重构矩阵
    matrices_exp = Q @ torch.diag_embed(L_exp) @ Q.transpose(-2, -1)

    # 提取 Voigt 分量
    exp_voigt = torch.stack([
        matrices_exp[:, 0, 0],
        matrices_exp[:, 1, 1],
        matrices_exp[:, 2, 2],
        matrices_exp[:, 1, 2],
        matrices_exp[:, 0, 2],
        matrices_exp[:, 0, 1]
    ], dim=1).to(orig_dtype)

    return exp_voigt, matrices_exp.to(orig_dtype)




def main():
    """Main training function."""
    config = {
        'hidden_dim': 48,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'num_epochs': 60,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'train_subset': 3000,  # Use a subset for quick testing; set to None for full dataset
        'lmax': 4,
        'num_workers': 4,
        'use_precomputed': True,  # Use precomputed graphs (much faster)
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    logger, experiment_name = setup_logger(config['save_dir'])

    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Experiment Name: {experiment_name}")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    logger.info("="*60)

    # Load data with precomputed or on-the-fly computation
    use_precomputed = config.get('use_precomputed', True)

    if use_precomputed:
        logger.info("Loading PRECOMPUTED graphs for ultra-fast training...")
    else:
        logger.info("Using ON-THE-FLY computation (slower)...")

    train_loader, val_loader, test_loader = get_dielectric_data_loaders(
        data_dir='data/mp_dielectric',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_subset=config['train_subset'],
    )

    # Get normalization constants (computed by the dataset)
    # Handle both precomputed and regular dataset structures
    if hasattr(train_loader.dataset, 'dataset'):
        # Subset case (e.g., when using train_subset)
        dataset = train_loader.dataset.dataset
    else:
        dataset = train_loader.dataset

    if hasattr(dataset, 'log_mean'):
        log_mean = dataset.log_mean
        log_std = dataset.log_std
        component_mean = dataset.component_mean  # [FIX] Component-wise normalization
        component_std = dataset.component_std
    else:
        # Fallback to default values
        log_mean = 2.4245  # Default from training data
        log_std = 0.5998
        component_mean = [log_mean]*3 + [0.0]*3
        component_std = [log_std]*6
        logger.warning("Using default normalization constants!")

    logger.info(f"Using normalization constants:")
    logger.info(f"  [Scalar] log_mean = {log_mean:.6f}, log_std = {log_std:.6f}")
    logger.info(f"  [Component-wise]")
    for i, name in enumerate(['ε11', 'ε22', 'ε33', 'ε23', 'ε13', 'ε12']):
        logger.info(f"    {name}: mean={component_mean[i]:.6f}, std={component_std[i]:.6f}")

    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)

    if use_precomputed:
        logger.info(f"  Mode: Using PRECOMPUTED graphs (ultra-fast!)")
    else:
        logger.info(f"  Mode: Using ON-THE-FLY computation (slower)")

    logger.info(f"  Training set: {len(train_loader.dataset)} structures")
    logger.info(f"  Validation set: {len(val_loader.dataset)} structures")
    logger.info(f"  Test set: {len(test_loader.dataset)} structures")

    if use_precomputed:
        logger.info("\n  [PERFORMANCE NOTES]")
        logger.info("  [+] No neighbor list computation during training")
        logger.info("  [+] No spherical harmonics computation")
        logger.info("  [+] No radial basis function computation")
        logger.info("  [+] Training speed will be significantly faster")
        logger.info("  [+] Epoch time will remain stable (no 'slowdown over time')")

    logger.info("="*60)

    # [FIX] Eigenvalue bounds - Single source of truth for initialization
    # After initialization, ALL code should read from model.min_log_eigenvalue/.max_log_eigenvalue
    MIN_EIGENVALUE = -0.8  # e^-0.8 ≈ 0.45 min variance (raised from -1.5 to prevent overconfidence)
    MAX_EIGENVALUE = 2.0    # e^2 ≈ 7.4 max variance (prevents explosion)

    # [FIX] Based on training log analysis, joint training causes variance collapse
    # Setting detach_uq_features=True prevents UQ branch from affecting feature extractor
    # Training log showed:
    #   - LogDet(Sigma): +1.18 → -12 (variance collapse)
    #   - Mahalanobis RMS: 1.4 → 5.3 (over-confidence increasing)
    #   - Uncertainty Loss: +0.17 → -1.87 (rewarding shrinkage)
    # You can set this to False to experiment with joint training
    DETACH_UQ_FEATURES = True  # Recommended: True for stability, False for joint training

    model = EquivariantUncertaintyNetwork(
        hidden_dim=config['hidden_dim'],
        max_radius=5.0,
        atom_feature_dim=49,
        lmax=config['lmax'],
        num_layers=2,
        covariance_scale=2.0,
        # Initialize with these bounds - stored as model attributes
        min_log_eigenvalue=MIN_EIGENVALUE,
        max_log_eigenvalue=MAX_EIGENVALUE,
        # Detach UQ branch to prevent variance collapse
        detach_uq_features=DETACH_UQ_FEATURES,
    ).to(config['device'])

    # Verify consistency between model and initialization
    assert model.min_log_eigenvalue == MIN_EIGENVALUE, "Model min eigenvalue mismatch!"
    assert model.max_log_eigenvalue == MAX_EIGENVALUE, "Model max eigenvalue mismatch!"
    logger.info(f"[+] Eigenvalue bounds: min={MIN_EIGENVALUE}, max={MAX_EIGENVALUE}")
    logger.info(f"[+] UQ Training Mode: {'Detached (feature protection)' if DETACH_UQ_FEATURES else 'Joint Training'}")

    # [修改] UQ 分支学习率调整
    # 特征网络和均值头：正常学习率
    # 协方差头（UQ分支）：使用相同学习率（已用 detach 保护，无需额外衰减）
    base_lr = config['learning_rate']
    cov_lr = base_lr  # 取消 0.1x 衰减，UQ 分支是独立优化任务

    # 分组参数
    feature_params = []
    cov_params = []

    # 定义属于协方差/UQ分支的关键词
    # A_base: 基础偏置
    # cov_head: 输出层
    # uq_bottleneck: UQ分支的中间层
    # sigma_pooling: UQ分支的池化层
    cov_keywords = ['cov_head', 'uq_bottleneck', 'A_base', 'sigma_pooling', 'cov_mlp']

    for name, param in model.named_parameters():
        # 只要参数名包含上述任何一个关键词，就归入 cov_params
        if any(k in name for k in cov_keywords):
            cov_params.append(param)
        else:
            feature_params.append(param)

    print(f"  Parameter groups (FIXED):")
    print(f"    Feature + Mean head: {len(feature_params)} parameters (lr={base_lr:.2e})")
    print(f"    Covariance (UQ branch): {len(cov_params)} parameters (lr={cov_lr:.2e})")
    print(f"  Training schedule (Multivariate Laplace NLL):")
    print(f"    Stage 1 (epochs 0-4): w_mse=0.1, w_nll=1.0 (brief warmup)")
    print(f"    Stage 2 (epochs 5+): w_mse=0.01, w_nll=1.0 (Laplacian NLL主导)")
    print(f"    → Laplacian NLL: D_M (linear) not D_M^2 (quadratic)")
    print(f"    → MAE-friendly loss, aligned with evaluation metric")
    print(f"    → min_eigenvalue=-1.5 prevents variance collapse")

    # 创建分组优化器
    optimizer = optim.AdamW([
        {'params': feature_params, 'lr': base_lr, 'weight_decay': 1e-3},
        {'params': cov_params, 'lr': cov_lr, 'weight_decay': 1e-4}  # 协方差头用更小的 weight_decay
    ], foreach=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=config['num_epochs'],
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e4
    )

    logger.info("\nUsing Sym^2(rho_c) covariance basis (21 matrices: 2×0e + 2×2e + 1×4e)")
    logger.info("\nTraining Configuration:")
    logger.info(f"  Peak Learning Rate: 1e-3 (OneCycleLR)")
    logger.info(f"  Training on {len(train_loader.dataset)} structures")
    logger.info("="*60)
    logger.info("\nStarting training...")

    # [修复] 在循环外创建 MetricCollection，防止内存泄漏
    # 直接放在 CPU 上，避免每个 epoch 都 to('cpu') 搬运
    # [FIX] Removed rmse (outlier-sensitive) and eigenvalue_mae (redundant with mae_diag)
    val_metrics_collection = MetricCollection({
        'mae': MeanAbsoluteError(),
        'mae_diag': MeanAbsoluteError(),
        'mae_off_diag': MeanAbsoluteError(),
        'spd_rate': SpdRate(),
        'anisotropy_mae': MeanAbsoluteError(),
    }).to('cpu')  # Metrics stay on CPU, no need to move every epoch

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mae_median': [],  # New: median MAE
        'val_log_mae': []      # New: log-space MAE
    }

    all_epoch_metrics = []
    best_val_mae = float('inf')
    best_val_nll = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_nll_epoch = 0
    nll_patience_counter = 0


    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        logger.info("-" * 60)

        train_loss, train_loss_components = train_epoch(
            model, train_loader, optimizer, config['device'],
            epoch=epoch, scheduler=scheduler
        )

        # 每5个epoch验证一次，加速训练
        should_validate = (epoch % 5 == 0) or (epoch == config['num_epochs'] - 1)

        if should_validate:
            # [修复] 传入预先创建的 metrics_collection
            val_metrics = validate(
                model, val_loader, config['device'],
                metrics_collection=val_metrics_collection,
                train_log_mean=log_mean,
                train_log_std=log_std
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            # [FIX] Removed val_rmse tracking - not useful due to outliers

            # Add new metrics to history
            if 'mae_median' in val_metrics:
                history['val_mae_median'].append(val_metrics['mae_median'])
            if 'log_mae' in val_metrics:
                history['val_log_mae'].append(val_metrics['log_mae'])

            epoch_metrics = log_epoch_metrics(
                logger, epoch, train_loss, val_metrics,
                optimizer.param_groups[0]['lr'], train_loss_components
            )
            all_epoch_metrics.append(epoch_metrics)

            # Save best MAE model (for point prediction accuracy)
            mae_improved = False
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                mae_improved = True
                best_epoch = epoch  # Update best_epoch for ALL epochs, not just epoch < 15
                patience_counter = 0  # Reset patience counter when MAE improves

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_metrics['mae'],
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'config': config,
                    'experiment_name': experiment_name,
                    # [FIX] Save component-wise normalization for consistency with test_and_visualize.py
                    'normalization': {
                        'log_mean': log_mean,
                        'log_std': log_std,
                        'component_mean': component_mean,  # [6] component-wise mean
                        'component_std': component_std      # [6] component-wise std
                    }
                }, os.path.join(config['save_dir'], 'best_mae.pth'))
                logger.info(f"[BEST MAE] New best MAE model saved! MAE: {best_val_mae:.4f}")

            # Save best NLL model (for uncertainty estimation validation)
            nll_improved = False
            if val_metrics['loss'] < best_val_nll:
                best_val_nll = val_metrics['loss']
                nll_improved = True
                if epoch >= 15:  # Track best NLL epoch after epoch 15
                    best_nll_epoch = epoch
                    nll_patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_metrics['mae'],
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'config': config,
                    'experiment_name': experiment_name,
                    # [FIX] Save component-wise normalization for consistency with test_and_visualize.py
                    'normalization': {
                        'log_mean': log_mean,
                        'log_std': log_std,
                        'component_mean': component_mean,  # [6] component-wise mean
                        'component_std': component_std      # [6] component-wise std
                    }
                }, os.path.join(config['save_dir'], 'best_nll.pth'))
                logger.info(f"[BEST NLL] New best NLL model saved! Loss: {best_val_nll:.4f}")

            # Update patience counters (for early stopping in Stage 3)
            if epoch >= 30:
                # Stage 3 (epoch >= 30): track NLL improvement for early stopping
                if not nll_improved:
                    nll_patience_counter += 1
            elif epoch >= 15:
                # Stage 2 (epoch 15-29): track NLL improvement but don't early stop
                if nll_improved:
                    best_nll_epoch = epoch
                    nll_patience_counter = 0

            # Check for early stopping (only in Stage 3: epoch >= 30)
            if epoch >= 30 and nll_patience_counter >= config['patience']:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                logger.info(f"Best NLL epoch: {best_nll_epoch + 1} with val_nll: {best_val_nll:.4f}")
                logger.info(f"Best MAE epoch: {best_epoch + 1} with val_mae: {best_val_mae:.4f}")
                logger.info(f"Stop reason: No NLL improvement for {nll_patience_counter}/{config['patience']} epochs (Current: {val_metrics['loss']:.4f}, Best: {best_val_nll:.4f})")
                break
        else:
            # 跳过验证的epoch，只记录训练loss
            # 不更新history中的val指标（保持同步）
            history['train_loss'].append(train_loss)
            # 延用上一次的val_metrics（如果存在）
            if 'val_metrics' in locals():
                history['val_loss'].append(val_metrics['loss'])
                history['val_mae'].append(val_metrics['mae'])
                # [FIX] Removed val_rmse tracking - not useful due to outliers
                if 'mae_median' in val_metrics:
                    history['val_mae_median'].append(val_metrics['mae_median'])
                if 'log_mae' in val_metrics:
                    history['val_log_mae'].append(val_metrics['log_mae'])

    logger.info("\n" + "-" * 60)
    logger.info("Training completed!")

    metrics_file = os.path.join(config['save_dir'], f"{experiment_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'config': config,
            'history': all_epoch_metrics,
            'best_epoch': best_epoch + 1,
            'best_val_mae': float(best_val_mae)
        }, f, indent=2)

    logger.info(f"\nAll metrics saved to: {metrics_file}")
    logger.info(f"Log file saved to: {os.path.join(config['save_dir'], f'{experiment_name}.log')}")

    with open(os.path.join(config['save_dir'], 'training_history.json'), 'w') as f:
        history_serializable = {}
        for k, v in history.items():
            history_serializable[k] = [float(x) for x in v]
        json.dump(history_serializable, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE (GPa)')
    plt.title('Validation MAE')
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.close()
    logger.info(f"\nResults saved to {config['save_dir']}")


if __name__ == "__main__":
    main()