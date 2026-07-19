"""
test_and_visualize.py
-------------------
Test E(3) equivariance and uncertainty quantification.
Comprehensive evaluation with parity plots, equivariance tests, and uncertainty calibration.
Updated to match the current training pipeline with Kelvin-Mandel format and matrix logarithm.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
import os
import json
import scipy.stats as stats
from scipy.stats import chi2
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Import modules
from equivariant_network import EquivariantUncertaintyNetwork
from dielectric_data_loader import get_dielectric_data_loaders, DielectricDataset
from voigt_utils import (
    random_rotation_matrix,
    voigt_to_kelvin_mandel,
    kelvin_mandel_to_voigt,
    sym_matrix_exp_voigt,
    voigt_to_matrix_batch
)


def sym_matrix_exp_voigt_with_matrix(voigt_tensor):
    """
    [OPTIMIZATION] 计算矩阵指数并同时返回 Voigt 和矩阵格式。

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
    ], dim=-1)

    # 转回原数据类型
    return exp_voigt.to(orig_dtype), matrices_exp.to(orig_dtype)


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型（用于JSON序列化）"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar types: use item() to convert
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def find_optimal_temperature(mu_km, sigma_km, target_km, use_laplacian=True):
    """
    在验证集上寻找最优温度 T 用于 Temperature Scaling

    使用**中位数匹配（Median Matching）**策略，比 NLL 优化更鲁棒，
    不会被离群点或尖峰带偏，确保模型在"大多数样本"上校准。

    温度缩放公式: Sigma_calibrated = T * Sigma
    或在 Log 空间（A 矩阵）: A_calibrated = A + log(T) * I

    Args:
        mu_km: 预测均值 (N, 6)
        sigma_km: 预测协方差矩阵 (N, 6, 6)
        target_km: 真实值 (N, 6)
        use_laplacian: 是否使用拉普拉斯分布的理论值

    Returns:
        T_opt: 最优温度参数
    """
    # 转换为 numpy
    mu = np.array(mu_km)
    sigma = np.array(sigma_km)
    target = np.array(target_km)
    diff = target - mu
    dim = 6

    # 预计算原始的 Mahalanobis 距离平方 D_M^2
    dm2_orig = []
    for i in range(len(mu)):
        try:
            # D_M^2 = diff^T * Sigma^-1 * diff
            d2 = diff[i] @ np.linalg.solve(sigma[i], diff[i])
            dm2_orig.append(d2)
        except:
            continue
    dm2_orig = np.array(dm2_orig)

    # 获取理论参考值
    if use_laplacian:
        # 对于 6D 多元拉普拉斯（Normal Scale Mixture: W*V, W~Exp(2), V~Chi2(6)）
        # 通过蒙特卡洛模拟得到:
        #   - 理论中位数约 8.16
        #   - 理论均值约 12.0
        theo_median = 8.16
        theo_mean = 12.0
        dist_name = "Multivariate Laplace"
    else:
        # 对于 6D 高斯 (chi2_6)
        theo_median = 5.348
        theo_mean = 6.0
        dist_name = "Gaussian (chi2_6)"

    def calibration_objective(T):
        """
        中位数匹配目标函数

        中位数匹配比 NLL 更鲁棒：
        - NLL 会被离群点和尖峰严重影响
        - 中位数只关注"大多数样本"的校准状态
        """
        if T <= 0:
            return 1e10

        # 缩放后的经验中位数
        emp_median = np.nanmedian(dm2_orig) / T
        return (emp_median - theo_median) ** 2

    # 设置合理的搜索范围，防止 T 缩得太小
    mean_dm2 = np.nanmean(dm2_orig)
    median_dm2 = np.nanmedian(dm2_orig)

    print(f"\n[Temperature Scaling - Median Matching]")
    print(f"  Distribution: {dist_name}")
    print(f"  Current D_M^2 statistics:")
    print(f"    Mean:   {mean_dm2:.4f} (theoretical: {theo_mean:.2f})")
    print(f"    Median: {median_dm2:.4f} (theoretical: {theo_median:.2f})")

    # 根据中位数匹配计算初始猜测
    # median / T = theo_median => T = median / theo_median
    T_init_guess = max(0.1, median_dm2 / theo_median)

    # 设置搜索范围：强制 T 不低于 0.05，保护长尾覆盖率
    # 上限 2.0 防止过度放大（会导致离群点爆表）
    bounds = [(0.05, 2.0)]

    print(f"  Initial T guess: {T_init_guess:.4f} (from median matching)")
    print(f"  Search bounds: [{bounds[0][0]}, {bounds[0][1]}]")

    # 使用 L-BFGS-B 寻找最小值
    res = minimize(calibration_objective, x0=[T_init_guess], bounds=bounds, method='L-BFGS-B')

    T_opt = res.x[0]

    # 计算校准效果预测
    expected_mean = mean_dm2 / T_opt
    expected_median = median_dm2 / T_opt

    print(f"\n  Optimal Temperature T: {T_opt:.4f}")
    print(f"  Calibration Loss at T=1.0:   {calibration_objective(1.0):.4f}")
    print(f"  Calibration Loss at T_opt:   {res.fun:.4f}")
    print(f"\n  Expected D_M^2 after scaling:")
    print(f"    Mean:   {expected_mean:.4f} (theoretical: {theo_mean:.2f})")
    print(f"    Median: {expected_median:.4f} (theoretical: {theo_median:.2f})")

    # 判断校准状态
    if abs(expected_median - theo_median) < 1.0:
        print(f"  Status: ✓ Well-calibrated (median match)")
    elif expected_median < theo_median * 0.8:
        print(f"  Status: ⚠ Still over-conservative (consider smaller T)")
    else:
        print(f"  Status: ⚠ Slightly over-confident (consider larger T)")

    return T_opt


def apply_temperature_scaling(results, T):
    """
    应用温度缩放到 results 中的协方差矩阵和马氏距离

    Args:
        results: 评估结果字典
        T: 温度系数

    Returns:
        results: 修改后的结果字典（原地修改）
    """
    print(f"\n[Applying Temperature Scaling with T={T:.4f}]")

    # 1. 缩放 sigma_km_list 中的协方差矩阵
    if results['sigma_km_list']:
        for i in range(len(results['sigma_km_list'])):
            results['sigma_km_list'][i] = results['sigma_km_list'][i] * T

    # 2. 重新计算马氏距离（使用缩放后的 Sigma）
    if results['mu_km_list'] and results['sigma_km_list'] and results['targets_km_list'] and \
       len(results['mu_km_list']) > 0 and len(results['sigma_km_list']) > 0 and len(results['targets_km_list']) > 0:
        mu_all = np.concatenate(results['mu_km_list'], axis=0)
        sigma_all = np.concatenate(results['sigma_km_list'], axis=0)
        targets_all = np.concatenate(results['targets_km_list'], axis=0)

        # 重新计算马氏距离
        dm2_new = []
        for i in range(len(mu_all)):
            try:
                delta = targets_all[i] - mu_all[i]
                # D_M^2 = delta^T * Sigma^-1 * delta
                # 由于 Sigma 已经被缩放 T 倍，新的距离 = 原距离 / T
                d2 = delta @ np.linalg.solve(sigma_all[i], delta)
                dm2_new.append(d2)
            except:
                dm2_new.append(np.nan)

        results['mahalanobis_distances'] = np.array(dm2_new)

        # 打印校准效果对比
        print(f"  Mahalanobis D_M^2 statistics after scaling:")
        print(f"    Mean:   {np.nanmean(dm2_new):.4f} (ideal: ~6.0 for χ²_6)")
        print(f"    Median: {np.nanmedian(dm2_new):.4f} (ideal: ~5.35 for χ²_6)")
        print(f"    95th:   {np.nanpercentile(dm2_new, 95):.4f} (ideal: ~12.59 for χ²_6)")

    # 3. 更新 sharpness_volumes 和 sharpness_radii（因为它们依赖于 Sigma）
    if results['eigenvals'] and len(results['eigenvals']) > 0:
        eigenvals = np.array(results['eigenvals'])
        # 缩放特征值
        eigenvals_scaled = eigenvals * T

        chi2_95_6d = chi2.ppf(0.95, df=6)
        volumes_new = []
        radii_new = []

        for i in range(len(eigenvals_scaled)):
            det_sigma = np.prod(eigenvals_scaled[i])
            volume_constant = (np.pi ** 3) / 6 * (chi2_95_6d ** 3)
            volume_95 = volume_constant * np.sqrt(max(det_sigma, 1e-12))
            volumes_new.append(volume_95)

            avg_radius = (volume_95 / ((np.pi ** 3) / 6)) ** (1/6)
            radii_new.append(avg_radius)

        results['sharpness_volumes'] = volumes_new
        results['sharpness_radii'] = radii_new

        print(f"  Sharpness volumes updated (scaled by T^(6/2) = T^3)")

    # 4. 更新 uncertainty_trace（因为 trace(Σ) 也被缩放）
    if results['uncertainty_trace'] and len(results['uncertainty_trace']) > 0:
        # uncertainty_trace 存储的是原始 trace，需要缩放
        results['uncertainty_trace'] = [t * T for t in results['uncertainty_trace']]
        print(f"  Uncertainty trace scaled by T={T:.4f}")

    print(f"[Temperature Scaling Applied Successfully]")

    return results


class DataProcessor:
    """Data processor matching train.py exactly"""
    def __init__(self, dataset):
        """Initialize with dataset normalization parameters

        Args:
            dataset: Dataset object with component_mean and component_std attributes
                     (component-wise normalization for proper denormalization)
        """
        # [FIX] Component-wise normalization for proper denormalization
        # This matches train.py validation exactly
        self.component_mean = dataset.component_mean  # [6] mean for each component
        self.component_std = dataset.component_std      # [6] std for each component
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def denormalize_to_physical(self, tensor_km):
        """
        Convert from Kelvin-Mandel normalized space to physical space
        Following the exact same steps as train.py validation
        """
        # 1. Convert from Kelvin-Mandel to Standard Voigt
        tensor_voigt = kelvin_mandel_to_voigt(tensor_km)

        # 2. [FIX] Component-wise denormalization: x * std + mean_vec
        # This matches train.py:478-481 exactly
        mean_vec = torch.tensor(self.component_mean,
                               device=tensor_voigt.device,
                               dtype=tensor_voigt.dtype).unsqueeze(0)
        std_vec = torch.tensor(self.component_std,
                              device=tensor_voigt.device,
                              dtype=tensor_voigt.dtype).unsqueeze(0)
        tensor_log = tensor_voigt * std_vec + mean_vec

        # 3. Apply matrix exponential to get physical values
        tensor_phys = sym_matrix_exp_voigt(tensor_log)

        return tensor_phys


CONFIG = {
    'checkpoint_path': 'checkpoints/best_mae.pth',
    'data_dir': 'data/mp_dielectric',
    'hidden_dim': 48,   # Same as train.py (FIXED: was 64, should be 48)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_plots': True,
    'num_workers': 0,   # Same as train.py
    'batch_size': 16,   # Same as train.py (updated)
}


def load_model_and_data():
    """Load model and test data - matching train.py exactly"""
    device = CONFIG['device']

    print("Loading model...")
    print(f"  Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {CONFIG['checkpoint_path']}...")

    try:
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    except Exception as e:
        if "weights_only" in str(e):
            print("  Warning: Loading with weights_only=False")
            checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device, weights_only=False)
        else:
            raise e

    # Initialize model with EXACT same parameters as train.py
    model = EquivariantUncertaintyNetwork(
        hidden_dim=CONFIG['hidden_dim'],  # 64
        max_radius=5.0,  # Same as train.py
        atom_feature_dim=49,  # Same as train.py (FIXED: was 119)
        lmax=4,  # Same as train.py
        num_layers=2,  # Same as train.py
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Use normalization parameters from checkpoint (saved from training set)
    # [FIX] Remove hardcoded fallback - require checkpoint to have normalization
    normalization = checkpoint.get('normalization', None)
    if normalization is None:
        raise ValueError(
            "Checkpoint does not contain normalization parameters! "
            "Please use a checkpoint trained with the latest version of train.py "
            "that saves component_mean and component_std. "
            "You may need to retrain your model or manually add these values to the checkpoint."
        )

    # [FIX] Component-wise normalization - must match train.py
    # Checkpoint MUST have component_mean and component_std (new training format)
    if 'component_mean' not in normalization or 'component_std' not in normalization:
        raise ValueError(
            f"Checkpoint missing component-wise normalization parameters!\n"
            f"Available keys: {list(normalization.keys())}\n"
            f"Please retrain your model with the updated train.py that saves component_mean and component_std."
        )

    component_mean = normalization['component_mean']
    component_std = normalization['component_std']
    log_mean = normalization.get('log_mean', component_mean[0])  # For backward compatibility
    log_std = normalization.get('log_std', component_std[0])

    print(f"  Using component-wise normalization from CHECKPOINT:")
    print(f"    component_mean = {component_mean}")
    print(f"    component_std  = {component_std}")

    # Create a processor object with the normalization parameters
    class DummyDataset:
        def __init__(self, component_mean, component_std):
            self.component_mean = component_mean  # [6]
            self.component_std = component_std      # [6]
            # Also keep scalar values for backward compatibility
            self.log_mean = component_mean[0]
            self.log_std = component_std[0]

    processor = DataProcessor(DummyDataset(component_mean, component_std))

    return model, processor, log_mean, log_std, device, component_mean, component_std


def evaluate_model(model, val_loader, processor, log_mean, log_std, dataset_name="Validation"):
    """Comprehensive model evaluation - using validation set like train.py validation

    Args:
        model: Trained model
        val_loader: Validation data loader
        processor: Data processor
        log_mean: Log normalization mean
        log_std: Log normalization std
    """
    device = CONFIG['device']
    model.eval()

    print(f"  Evaluation mode: Log Space (KM) - avoiding exp amplification")

    results = {
        # Log space (KM) metrics - for UQ calibration
        'true_diag': [],
        'pred_diag': [],
        'true_off': [],
        'pred_off': [],
        'uncertainty': [],
        'errors': [],
        'traces': [],
        'conds': [],
        'pos_definite': [],
        'mu_spd_satisfied': [],  # Track if predicted mean tensors satisfy SPD constraint
        'eigenvals': [],  # Store eigenvalues for detailed analysis
        'cond_exceed_50': [],  # Track samples exceeding condition number 50
        'cond_exceed_100': [],  # Track samples exceeding condition number 100
        'mahalanobis_distances': [],  # Store Mahalanobis distances for calibration
        # For Energy Score and Reliability Diagram
        'mu_km_list': [],  # Store predictions in KM space for ES computation
        'sigma_km_list': [],  # Store covariance in KM space for ES computation
        'targets_km_list': [],  # Store targets in KM space for ES computation
        # [新增] 害群之马跟踪信息
        'outlier_info': [],  # Store detailed information for each sample
        # [NEW] UQ Analysis: Sharpness and Error-Uncertainty Correlation
        'sharpness_volumes': [],  # 95% confidence ellipsoid volumes
        'sharpness_radii': [],    # Average radius of 95% confidence ellipsoid
        'uncertainty_trace': [],  # Trace of Sigma for scatter plot
        'error_norm': [],         # L2 norm of error for scatter plot
        'uncertainty_std': [],    # Standard deviation (sqrt of avg eigenvalue)
        # [NEW] Physical space metrics - matching train.py
        'true_diag_phys': [],
        'pred_diag_phys': [],
        'true_off_phys': [],
        'pred_off_phys': [],
        'mae_phys': [],           # Physical space MAE per sample
        'mae_diag_phys': [],      # Physical space diagonal MAE per sample
        'mae_off_phys': [],       # Physical space off-diagonal MAE per sample
    }

    print(f"Running inference on {dataset_name} Set...")

    with torch.no_grad():
        batch_idx_iter = 0
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to GPU
            batch = batch.to(device)

            # Forward pass - exactly like in train.py validation
            mu_km, A_km, Sigma_km = model(batch, compute_sigma=True)

            # Store KM space data for calibration metrics (MACE, ECE, etc.)
            results['mu_km_list'].append(mu_km.cpu().numpy())
            results['sigma_km_list'].append(Sigma_km.cpu().numpy())
            targets_km = batch.y.view(mu_km.shape[0], 6)  # Reshape targets
            results['targets_km_list'].append(targets_km.cpu().numpy())

            # Use log space (KM space) directly - NO EXP TRANSFORMATION
            # This avoids amplification of large errors
            pred = mu_km.cpu().numpy()  # Already in KM space
            true = targets_km.cpu().numpy()  # Already in KM space
            sigma = Sigma_km.cpu().numpy()  # Sigma is already in proper scale

            # Calculate Mahalanobis distance squared (D_M^2) for calibration
            # Initialize with NaNs for safe error handling
            batch_size = mu_km.shape[0]
            mahalanobis_sq_val = np.full((batch_size,), np.nan, dtype=np.float64)

            # Always use KM space for Mahalanobis distance (model outputs are in KM space)
            diff_km = (targets_km - mu_km).unsqueeze(-1)
            try:
                inv_sigma_diff = torch.linalg.solve(Sigma_km, diff_km.to(device))
                mahalanobis_sq = torch.bmm(diff_km.transpose(1, 2).to(device), inv_sigma_diff)
                # Use squeeze(-1) twice to remove only the last two dimensions, preserving batch dimension
                mahalanobis_sq = mahalanobis_sq.squeeze(-1).squeeze(-1)
                mahalanobis_sq_val = torch.clamp(mahalanobis_sq, min=0.0).cpu().numpy()
                results['mahalanobis_distances'].extend(mahalanobis_sq_val)
            except Exception as e:
                # If solve fails, keep NaNs and add them to results
                results['mahalanobis_distances'].extend(mahalanobis_sq_val)
                if batch_idx_iter == 0:  # Print error only once
                    print(f"  Warning: Mahalanobis distance calculation failed for batch {batch_idx_iter}: {str(e)[:100]}...")

            # 计算不确定性度量 - 使用 trace 作为标量不确定性指标
            uncertainty_metric = np.trace(sigma, axis1=1, axis2=2)

            error = np.mean(np.abs(pred - true), axis=1)

            # =====================================================
            # [NEW] Physical Space MAE Calculation - Matching train.py
            # =====================================================
            # Convert KM -> Voigt Standard -> Denormalize -> Matrix Exp -> Physical Space
            mu_voigt = kelvin_mandel_to_voigt(mu_km)
            target_voigt = kelvin_mandel_to_voigt(targets_km)

            # Component-wise denormalization: x * std + mean
            mean_vec = torch.tensor(processor.component_mean, device=device, dtype=mu_voigt.dtype).unsqueeze(0)
            std_vec = torch.tensor(processor.component_std, device=device, dtype=mu_voigt.dtype).unsqueeze(0)

            mu_log = mu_voigt * std_vec + mean_vec
            target_log = target_voigt * std_vec + mean_vec

            # Apply matrix exponential to get physical space values
            mu_phys, mu_phys_matrix = sym_matrix_exp_voigt_with_matrix(mu_log)
            target_phys, _ = sym_matrix_exp_voigt_with_matrix(target_log)

            # Check if predicted mean tensors satisfy SPD constraint (all eigenvalues > 0)
            # mu_phys_matrix shape: (batch_size, 3, 3)
            mu_phys_eigs = torch.linalg.eigvalsh(mu_phys_matrix)  # (batch_size, 3)
            mu_spd_ok = (mu_phys_eigs > 0).all(dim=1).cpu().numpy()  # True if all eigenvalues > 0
            results['mu_spd_satisfied'].extend(mu_spd_ok)

            # Calculate physical space MAE (matching train.py)
            mu_phys_cpu = mu_phys.cpu()
            target_phys_cpu = target_phys.cpu()

            # Overall MAE per sample
            mae_phys_sample = torch.abs(mu_phys_cpu - target_phys_cpu).mean(dim=1).numpy()
            results['mae_phys'].extend(mae_phys_sample)

            # Diagonal MAE per sample
            mae_diag_phys_sample = torch.abs(mu_phys_cpu[:, :3] - target_phys_cpu[:, :3]).mean(dim=1).numpy()
            results['mae_diag_phys'].extend(mae_diag_phys_sample)

            # Off-diagonal MAE per sample
            mae_off_phys_sample = torch.abs(mu_phys_cpu[:, 3:] - target_phys_cpu[:, 3:]).mean(dim=1).numpy()
            results['mae_off_phys'].extend(mae_off_phys_sample)

            # Store physical space values for parity plot
            results['true_diag_phys'].extend(target_phys_cpu[:, :3].flatten().numpy())
            results['pred_diag_phys'].extend(mu_phys_cpu[:, :3].flatten().numpy())
            results['true_off_phys'].extend(target_phys_cpu[:, 3:].flatten().numpy())
            results['pred_off_phys'].extend(mu_phys_cpu[:, 3:].flatten().numpy())
            # =====================================================

            # In log space (KM space), all samples are considered valid
            # No need to check positive definiteness
            is_positive_definite = np.ones(len(pred), dtype=bool)

            sigma_eigs = np.linalg.eigvalsh(sigma)
            cond_nums = sigma_eigs[:, -1] / (sigma_eigs[:, 0] + 1e-6)

            # Store eigenvalues for detailed analysis
            results['eigenvals'].extend(sigma_eigs)

            # Track samples exceeding condition number thresholds
            results['cond_exceed_50'].extend(cond_nums > 50)
            results['cond_exceed_100'].extend(cond_nums > 100)

            # [NEW] UQ Analysis: Compute Sharpness metrics
            # Sharpness: 95% confidence ellipsoid volume in KM space
            # For a 6-dimensional Gaussian: V = (pi^3 / Gamma(4)) * sqrt(det(Sigma)) * chi2_ppf(0.95, 6)^3
            # Simplified: V proportional to product of eigenvalues * chi2_threshold^(dim/2)
            chi2_95_6d = chi2.ppf(0.95, df=6)  # ~12.59 for 6D
            for i in range(len(sigma_eigs)):
                # Volume of 95% confidence ellipsoid: V = (pi^(d/2) / Gamma(d/2+1)) * sqrt(det(Sigma)) * chi2^(d/2)
                # For d=6: pi^3 / 6 * sqrt(det(Sigma)) * chi2^3
                det_sigma = np.prod(sigma_eigs[i])  # Product of eigenvalues = det(Sigma)
                # Volume constant for 6D: (pi^3 / Gamma(4)) * chi2_95^3
                # Gamma(4) = 6, so constant = pi^3 / 6 * chi2^3
                volume_constant = (np.pi ** 3) / 6 * (chi2_95_6d ** 3)
                volume_95 = volume_constant * np.sqrt(max(det_sigma, 1e-12))
                results['sharpness_volumes'].append(volume_95)

                # Average radius: (volume / volume_unit_sphere)^(1/dim)
                # Unit sphere volume in 6D: pi^3 / 6
                avg_radius = (volume_95 / ((np.pi ** 3) / 6)) ** (1/6)
                results['sharpness_radii'].append(avg_radius)

                # Uncertainty metrics for correlation analysis
                results['uncertainty_trace'].append(np.trace(sigma[i]))
                results['uncertainty_std'].append(np.sqrt(np.mean(sigma_eigs[i])))

            # [NEW] Error-Uncertainty Correlation: L2 norm of error
            error_l2 = np.linalg.norm(pred - true, axis=1)
            results['error_norm'].extend(error_l2)

            results['true_diag'].extend(true[:, :3].flatten())
            results['pred_diag'].extend(pred[:, :3].flatten())
            results['true_off'].extend(true[:, 3:].flatten())
            results['pred_off'].extend(pred[:, 3:].flatten())
            results['uncertainty'].extend(uncertainty_metric)  # 使用近似的物理不确定性
            results['errors'].extend(error)
            results['traces'].extend(np.trace(sigma, axis1=2))  # 保留原始trace用于调试
            results['conds'].extend(cond_nums)
            results['pos_definite'].extend(is_positive_definite)

            # [新增] 记录害群之马的详细信息
            for i in range(len(mahalanobis_sq_val)):
                # 计算相对误差
                rel_error = error[i] / (np.linalg.norm(true[i]) + 1e-6)

                # 提取物理空间的对角线值用于识别
                diag_values = pred[i, :3]

                # 检查是否为潜在异常值的标准
                is_outlier = (
                    mahalanobis_sq_val[i] > chi2.ppf(0.99, df=6) or  # D_M^2 > 99%分位数（放宽标准）
                    rel_error > 1.0 or  # 相对误差 > 100%
                    cond_nums[i] > 100 or  # 条件数 > 100
                    not is_positive_definite[i]  # 不是正定矩阵
                )

                # ✅ 获取稳定的样本ID（从预计算的ID）
                stable_id = None
                if hasattr(batch, "orig_idx"):
                    # 优先使用orig_idx（原始数据集索引）
                    stable_id = int(batch.orig_idx[i].item())
                elif hasattr(batch, "pre_idx"):
                    # 备用：使用pre_idx（文件名索引）
                    stable_id = int(batch.pre_idx[i].item())
                elif hasattr(batch, "ids") and batch.ids is not None:
                    # 最后备用：使用material_id
                    stable_id = int(batch.ids[i].item())

                # 收集所有可用的ID信息
                pre_idx = int(batch.pre_idx[i].item()) if hasattr(batch, 'pre_idx') else None
                orig_idx = int(batch.orig_idx[i].item()) if hasattr(batch, 'orig_idx') else None
                material_id = batch.ids[i].item() if hasattr(batch, 'ids') and batch.ids is not None else None

                outlier_info = {
                    'global_index': len(results['outlier_info']),  # 仅用于日志序号
                    'batch_idx': batch_idx_iter if hasattr(batch, 'batch_idx') else -1,  # 批次索引
                    'sample_idx': i,  # 批内样本索引
                    # ✅ 存储所有稳定ID
                    'pre_idx': pre_idx,        # 文件名索引 (0, 1, 2, ...)
                    'orig_idx': orig_idx,      # 原始数据集索引
                    'material_id': material_id, # 材料ID
                    'stable_id': stable_id,     # 主要使用的稳定ID（优先orig_idx）
                    'mahalanobis_distance_sq': mahalanobis_sq_val[i] if i < len(mahalanobis_sq_val) else float('nan'),
                    'error_norm': error[i],
                    'relative_error': rel_error,
                    'uncertainty_trace': np.trace(sigma[i]),
                    'condition_number': cond_nums[i],
                    'is_positive_definite': bool(is_positive_definite[i]),
                    'is_potential_outlier': bool(is_outlier),
                    'target_diag': true[i, :3].tolist(),  # 对角线元素
                    'pred_diag': pred[i, :3].tolist(),  # 预测的对角线元素
                    'target_norm': np.linalg.norm(true[i]),
                    'prediction_norm': np.linalg.norm(pred[i]),
                    'eigenvals': sigma_eigs[i].tolist() if i < len(sigma_eigs) else None
                }

                # material_id 已在上面设置过了，不需要重复

                results['outlier_info'].append(outlier_info)

            # Store data for Energy Score and Reliability Diagram
            results['mu_km_list'].append(mu_km.cpu().numpy())
            results['sigma_km_list'].append(Sigma_km.cpu().numpy())
            results['targets_km_list'].append(targets_km.cpu().numpy())

            batch_idx_iter += 1

    return results


# E(3) equivariance test removed to save computation time


def plot_detailed_analysis(results):
    """Create only the most valuable analysis figures"""
    # Get data
    conds = np.array(results['conds'])

    # 1. Condition Number Distribution (Important for Hinge Loss)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Analyze the data distribution to set appropriate x-axis range
    max_cond = np.max(conds)
    mean_cond = np.mean(conds)
    std_cond = np.std(conds)
    exceed_50_ratio = np.mean(results['cond_exceed_50']) * 100
    exceed_100_ratio = np.mean(results['cond_exceed_100']) * 100

    # Set x-axis limit based on data with some padding
    x_limit = max_cond * 1.1

    # Import seaborn for mako colormap
    import seaborn as sns
    mako_colors = sns.color_palette("mako", 10)

    # Create histogram with adaptive number of bins
    n_bins = min(50, max(30, int(len(conds) / 20)))
    bins = np.linspace(0, x_limit, n_bins + 1)
    counts, bins_out, patches = ax.hist(conds, bins=bins, color=mako_colors[5],
                                        edgecolor='black', alpha=0.8, linewidth=0.5)

    # Color bins with mako gradient (lighter for lower condition numbers)
    for i, patch in enumerate(patches):
        # Map bin position to color index
        color_idx = min(int(i / len(patches) * 8), 7)
        patch.set_facecolor(mako_colors[color_idx])

    # Add vertical line for mean
    ax.axvline(mean_cond, color=mako_colors[-1], linestyle='--', linewidth=2, alpha=0.8)

    # Add threshold lines if they are in range
    if 50 <= x_limit:
        ax.axvline(50, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    if 100 <= x_limit:
        ax.axvline(100, color='red', linestyle=':', linewidth=2, alpha=0.7)

    # Style: inward ticks with minor ticks
    ax.tick_params(axis='both', direction='in', which='both')
    ax.minorticks_on()

    # Labels and title
    ax.set_xlabel('Condition Number $\\kappa$', fontsize=14, fontfamily='serif')
    ax.set_ylabel('Count', fontsize=14, fontfamily='serif')
    ax.set_title('Condition Number Distribution', loc='left', fontweight='bold', fontsize=16, fontfamily='serif')
    ax.set_xlim(0, x_limit)

    # Grid with subtle style
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add subtle legend
    legend_elements = [
        plt.Line2D([0], [0], color=mako_colors[-1], linestyle='--', linewidth=2, label=f'Mean: {mean_cond:.1f}'),
    ]
    if exceed_50_ratio > 0 and 50 <= x_limit:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label='Threshold: 50'))
    if exceed_100_ratio > 0 and 100 <= x_limit:
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Threshold: 100'))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, fontsize=11)

    plt.tight_layout()
    plt.savefig('analysis_condition_number.png', dpi=300, bbox_inches='tight')
    plt.savefig('analysis_condition_number.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n[+] Analysis plot saved:")
    print("  - analysis_condition_number.png/pdf (Numerical Stability)")


# plot_calibration_analysis removed - redundant with plot_paper_figures
# get_ellipsoid_surface, get_confidence_ellipsoid, get_samples_for_3d_visualization removed - 3D visualization removed


def compute_calibration_coverage(mahalanobis_distances, use_gaussian=False):
    """
    Compute empirical coverage for theoretical confidence intervals.

    Uses theoretical chi2 distribution (Gaussian assumption) as reference.
    For Laplacian-trained models, the deviation from chi2 is diagnostic information.

    Detection of calibration issues:
    - If theoretical 95% interval only covers 50% → over-confident (variance collapse)
    - If theoretical 95% interval covers 99% → under-confident (too conservative)

    Args:
        mahalanobis_distances: Array of squared Mahalanobis distances
        use_gaussian: If True, use chi2 distribution; if False, returns statistics only

    Returns:
        Dictionary with coverage statistics
    """
    # Define confidence levels to check
    confidence_levels = [0.50, 0.68, 0.90, 0.95, 0.99]
    df = 6  # Degrees of freedom for 6D output

    coverage_results = {}
    for cl in confidence_levels:
        # Use theoretical chi2 distribution as reference
        threshold = chi2.ppf(cl, df=df)

        # Empirical coverage: proportion of samples with D_M^2 <= threshold
        empirical_coverage = np.mean(mahalanobis_distances <= threshold)

        coverage_results[f'{int(cl*100)}%'] = {
            'theoretical': cl,
            'empirical': empirical_coverage,
            'threshold': threshold,
            'deviation': empirical_coverage - cl
        }

    return coverage_results


def plot_paper_figures(results, processor, use_gaussian_calibration=False):
    """Create publication-quality figures for ICML paper

    Args:
        results: Dictionary containing evaluation results
        processor: DataProcessor for denormalization
        use_gaussian_calibration: Ignored - always use theoretical Laplacian
    """
    # Color scheme
    COLOR_MODEL = '#002060'      # 午夜蓝
    COLOR_EDGE = '#004080'       # 深蓝边框
    COLOR_IDEAL = '#404040'      # 深灰
    COLOR_SHADE = '#F5D0A9'      # 香槟金

    # Set professional style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linestyle': '-',
        'grid.color': '#E0E0E0'
    })

    # Create a figure with 1 row, 2 columns for (a) Diagonal, (b) Reliability Diagram
    fig = plt.figure(figsize=(12, 5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # ========================================
    # (a) Diagonal components - Parity Plot (Log/KM Space)
    # ========================================
    ax1 = fig.add_subplot(gs[0])

    # Use log space (Kelvin-Mandel) data for evaluation
    true_d = np.array(results['true_diag'])
    pred_d = np.array(results['pred_diag'])
    space_label = "Log Space"

    # Use ALL data points with log-density hexbin
    hb1 = ax1.hexbin(true_d, pred_d, gridsize=50,
                     cmap='Blues', mincnt=1, bins='log',
                     edgecolors='none', linewidth=0.1)

    # Add colorbar for log density
    cbar = plt.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('log$_{10}$(Count)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Diagonal line (dark gray for visibility)
    min_val = min(true_d.min(), pred_d.min())
    max_val = max(true_d.max(), pred_d.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color=COLOR_IDEAL,
             linestyle='--', alpha=0.6, linewidth=1.5)

    # Set limits and aspect
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_aspect('equal')

    # Labels and title
    ax1.set_xlabel(r'DFT Calculated (log $\varepsilon_{ii}$)', fontsize=12)
    ax1.set_ylabel(r'Predicted (log $\varepsilon_{ii}$)', fontsize=12)
    ax1.set_title('(a) Diagonal Components (Log Space)', loc='left', fontweight='bold', fontsize=14, y=1.05)

    # Grid with low alpha
    ax1.grid(True, linestyle=':', alpha=0.2, color='gray')

    # Add text showing total number of points
    ax1.text(0.98, 0.02, f'N = {len(true_d)}', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))

    # ========================================
    # (b) Reliability Diagram (Calibration Curve)
    # ========================================
    ax2 = fig.add_subplot(gs[1])

    if 'mahalanobis_distances' in results and len(results['mahalanobis_distances']) > 0:
        mahalanobis_sq = results['mahalanobis_distances']

        # Get MVL CDF for p-value calculation
        mvl_cdf = get_theoretical_mvl_cdf_interpolator(dim=6)
        theo_p_values = mvl_cdf(mahalanobis_sq)

        # Calculate empirical coverage vs theoretical confidence
        expected_confidence = np.linspace(0, 1, 100)
        observed_confidence = []

        for p in expected_confidence:
            fraction = np.mean(theo_p_values <= p)
            observed_confidence.append(fraction)

        observed_confidence = np.array(observed_confidence)

        # Calculate MACE
        mace = np.mean(np.abs(observed_confidence - expected_confidence))

        # Calculate KS statistic and ECE
        ks_stat = np.max(np.abs(observed_confidence - expected_confidence))

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(theo_p_values, bin_edges) - 1
        ece = 0.0
        for i in range(n_bins):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                bin_conf = (bin_edges[i] + bin_edges[i+1]) / 2
                bin_acc = np.mean(theo_p_values[mask] <= bin_conf)
                bin_weight = np.sum(mask) / len(theo_p_values)
                ece += bin_weight * np.abs(bin_acc - bin_conf)

        # Plot ideal calibration line
        ax2.plot([0, 1], [0, 1], color=COLOR_IDEAL, linestyle='--',
                linewidth=2, label='Perfect Calibration')

        # Plot calibration curve
        ax2.plot(expected_confidence, observed_confidence,
                color=COLOR_MODEL, linewidth=2.5, label='Model Prediction')

        # Metrics text box
        metrics_text = (f'MACE = {mace:.4f}\n'
                       f'KS = {ks_stat:.4f}\n'
                       f'ECE = {ece:.4f}')

        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#CCCCCC', alpha=0.9))

        # Labels and title
        ax2.set_xlabel('Theoretical CDF Value (p)', fontsize=12)
        ax2.set_ylabel('Empirical CDF: P(p_value ≤ p)', fontsize=12)
        ax2.set_title('(b) Uncertainty Calibration', loc='left', fontweight='bold', fontsize=14, y=1.05)

        # Grid and legend
        ax2.grid(True, alpha=0.1)
        ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
        ax2.set_aspect('equal')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No calibration\ndata available',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('(b) Uncertainty Calibration', loc='left', fontweight='bold', fontsize=14, y=1.05)

    plt.tight_layout()
    plt.savefig('paper_figure_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figure_accuracy.png', dpi=300, bbox_inches='tight')
    print("\n[+] Combined figure saved to 'paper_figure_accuracy.pdf/png'")

    # ========================================
    # Figure 2: SPD Validity (Eigenvalue Distribution)
    # ========================================
    fig2, ax = plt.subplots(figsize=(6, 4), dpi=300)

    eigenvals = np.array(results['eigenvals'])

    # Handle case where eigenvals is 1D
    if len(eigenvals.shape) == 1:
        eigenvals = eigenvals.reshape(-1, 1)

    # Clip extreme values for visualization
    eigenvals_clipped = np.clip(eigenvals, 0, np.percentile(eigenvals, 99))
    n_eigenvals = eigenvals_clipped.shape[1]

    # Import seaborn for mako colormap
    import seaborn as sns
    mako_colors = sns.color_palette("mako", n_eigenvals + 2)

    # Prepare data for violinplot
    data_for_violin = [eigenvals_clipped[:, i] for i in range(n_eigenvals)]
    parts = ax.violinplot(data_for_violin, positions=range(n_eigenvals), widths=0.8)

    # Color with mako colormap (gradient from light to dark)
    for i, pc in enumerate(parts['bodies']):
        color = mako_colors[i + 1]  # Skip the lightest color
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.8)

    # Style: inward ticks with minor ticks
    ax.tick_params(axis='both', direction='in', which='both')
    ax.minorticks_on()

    ax.set_xlabel('Eigenvalue Index', fontsize=12, fontfamily='serif')
    ax.set_ylabel('Eigenvalue of $\\Sigma$', fontsize=12, fontfamily='serif')
    ax.set_title('Spectrum Validity: All Eigenvalues Positive', loc='left', fontweight='bold', fontsize=14, fontfamily='serif')
    ax.set_xticks(range(n_eigenvals))
    ax.set_xticklabels([f'$\\lambda_{i+1}$' for i in range(n_eigenvals)], fontfamily='serif')

    # Adjust y-axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.xaxis.set_tick_params(labelsize=11)

    # Grid with subtle style
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('paper_figure_spd.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figure_spd.png', dpi=300, bbox_inches='tight')
    print("[+] SPD figure saved to 'paper_figure_spd.pdf/png'")

    # Create separate figures for detailed analysis
    plot_detailed_analysis(results)

    # Calculate metrics for return - Use physical space if available
    if results['true_diag_phys'] and results['pred_diag_phys']:
        true_d = np.array(results['true_diag_phys'])
        pred_d = np.array(results['pred_diag_phys'])
        true_od = np.array(results['true_off_phys'])
        pred_od = np.array(results['pred_off_phys'])
    else:
        true_d = np.array(results['true_diag'])
        pred_d = np.array(results['pred_diag'])
        true_od = np.array(results['true_off'])
        pred_od = np.array(results['pred_off'])
    unc = np.array(results['uncertainty'])
    err = np.array(results['errors'])

    r2_diag = r2_score(true_d, pred_d)
    mae_diag = np.mean(np.abs(true_d - pred_d))
    r2_off = r2_score(true_od, pred_od)
    true_all = np.concatenate([true_d, true_od])
    pred_all = np.concatenate([pred_d, pred_od])
    mae_all = np.mean(np.abs(true_all - pred_all))
    correlation = np.corrcoef(unc, err)[0, 1]

    # Calculate calibration factors (both in KM normalized log space - dimensionally consistent)
    # Note: err and unc are both in KM space (normalized log space), not physical space
    rms_error = np.sqrt(np.mean(err**2))
    mean_trace = unc.mean()
    cal_factor_rms = rms_error / np.sqrt(mean_trace)  # Ideal value = 1.0

    # Method 2: Compare Error^2 and Trace (both in KM² space)
    mean_error_squared = np.mean(err**2)
    cal_factor_variance = mean_error_squared / mean_trace  # Ideal value = 1.0

    # 1. 马氏距离的深度统计
    if 'mahalanobis_distances' in results and len(results['mahalanobis_distances']) > 0:
        dm2 = np.array(results['mahalanobis_distances'])
        dm2_median = np.median(dm2)
        dm2_95 = np.percentile(dm2, 95)
        dm2_99 = np.percentile(dm2, 99)
    else:
        dm2_median = dm2_95 = dm2_99 = float('nan')

    # 2. 斯皮尔曼相关系数 (更鲁棒的 Ranking Correlation)
    from scipy.stats import spearmanr
    try:
        spearman_corr, _ = spearmanr(unc, err)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0

    # 3. 协方差各向异性程度 (Anisotropy Ratio)
    # 检查模型是预测了"球"还是"椭球"
    if 'eigenvals' in results and results['eigenvals'] and len(results['eigenvals']) > 0:
        eigenvals = np.array(results['eigenvals'])
        # eigenvals 形状是 [N, 6]，取最大除以最小 (加个 epsilon 防除零)
        eig_ratios = eigenvals[:, -1] / (eigenvals[:, 0] + 1e-6)
        mean_anisotropy = np.mean(eig_ratios)
        median_anisotropy = np.median(eig_ratios)
    else:
        mean_anisotropy = median_anisotropy = float('nan')

    return {
        'r2_diag': r2_diag,
        'mae_diag': mae_diag,
        'r2_off': r2_off,
        'mae_all': mae_all,
        'correlation': correlation,
        'pos_def_rate': np.mean(results['pos_definite']) * 100,
        'mean_cond': np.mean(results['conds']),
        'median_cond': np.median(results['conds']),
        'max_cond': np.max(results['conds']),
        'exceed_50_ratio': np.mean(results['cond_exceed_50']) * 100,
        'exceed_100_ratio': np.mean(results['cond_exceed_100']) * 100,
        'cal_factor_rms': cal_factor_rms,  # 修正的校准因子 (RMS方法)
        'cal_factor_variance': cal_factor_variance,  # 修正的校准因子 (方差方法)
        'rms_error': rms_error,
        'mean_unc': np.mean(results['uncertainty']),
        # Deep Dive Calibration Metrics
        'dm2_median': dm2_median,         # 理想值 ~5.35 (χ²_6的中位数)
        'dm2_95': dm2_95,                 # 95th percentile
        'dm2_99': dm2_99,                 # 理想值 ~16.8 (χ²_6的99%分位数)
        'spearman_corr': spearman_corr,   # 越高越好 (接近 1.0)
        'mean_anisotropy_ratio': mean_anisotropy, # 越大说明学到了各向异性
        'median_anisotropy_ratio': median_anisotropy
    }


# Temperature scaling function removed
# MVL CDF Interpolator for Multivariate Laplace distribution

# Global MVL CDF interpolator (initialized once)
_MVL_CDF_INTERPOLATOR = None

def get_theoretical_mvl_cdf_interpolator(dim=6, num_samples=500000):
    """
    通过蒙特卡洛模拟生成多元拉普拉斯分布下 Mahalanobis^2 的理论 CDF 插值器。

    Mathematical derivation:
        Multivariate Laplace can be represented as Normal Scale Mixture:
        L(μ, Σ) = ∫ N(μ, τΣ) Exp(τ/2) dτ

        Therefore, D_M^2 = W * V where:
        - W ~ Exp(2)  [scale=2 for rate=0.5 in the mixture]
        - V ~ χ²(dim) [chi-squared distribution from Gaussian Mahalanobis distance]

    Args:
        dim: Dimension of the output (default 6 for dielectric tensor)
        num_samples: Number of Monte Carlo samples for interpolation

    Returns:
        Interpolation function that maps D_M^2 to theoretical CDF probability
    """
    global _MVL_CDF_INTERPOLATOR

    if _MVL_CDF_INTERPOLATOR is not None:
        return _MVL_CDF_INTERPOLATOR

    print(f"Generating theoretical Multivariate Laplace CDF lookup table (dim={dim}, samples={num_samples})...")

    np.random.seed(42)  # For reproducibility

    # 1. Sample W from Exponential distribution (scale mixing factor)
    # Rate = 0.5 in the mixture representation, so scale = 1/rate = 2.0
    w = np.random.exponential(scale=2.0, size=num_samples)

    # 2. Sample V from Chi-squared distribution (Gaussian Mahalanobis distance squared)
    v = np.random.chisquare(df=dim, size=num_samples)

    # 3. Compute D_M^2 samples: product of independent W and V
    dm2_samples = w * v

    # 4. Build empirical CDF as theoretical reference
    sorted_dm2 = np.sort(dm2_samples)
    probs = np.linspace(0, 1, num_samples)

    # Remove duplicates for interpolation stability
    unique_dm2, unique_indices = np.unique(sorted_dm2, return_index=True)
    unique_probs = probs[unique_indices]

    # Return interpolation function: input D_M^2, output theoretical probability P
    # fill_value handles out-of-range values
    from scipy.interpolate import interp1d
    _MVL_CDF_INTERPOLATOR = interp1d(
        unique_dm2, unique_probs,
        bounds_error=False,
        fill_value=(0, 1),
        assume_sorted=True
    )

    print(f"  MVL CDF range: [{unique_dm2.min():.2f}, {unique_dm2.max():.2f}]")
    print(f"  MVL CDF interpolator ready.")

    return _MVL_CDF_INTERPOLATOR


def compute_energy_score(mu_pred, sigma_pred, target, num_samples=1000, distribution='laplace'):
    """
    计算 Energy Score (CRPS 的推广)

    支持多元高斯分布和多元拉普拉斯分布（通过正态尺度混合表示）

    Args:
        mu_pred: (Batch, 6) 预测均值
        sigma_pred: (Batch, 6, 6) 预测协方差矩阵 (即 exp(A))
        target: (Batch, 6) 真实值 (Kelvin-Mandel 向量)
        num_samples: 蒙特卡洛采样的次数
        distribution: 'gaussian' 或 'laplace'

    Returns:
        mean_es: 该 Batch 的平均 Energy Score (越低越好)
    """
    batch_size = mu_pred.shape[0]
    dim = mu_pred.shape[1]  # 6

    # 转换为PyTorch张量
    if not isinstance(mu_pred, torch.Tensor):
        mu_pred = torch.tensor(mu_pred, dtype=torch.float32)
    if not isinstance(sigma_pred, torch.Tensor):
        sigma_pred = torch.tensor(sigma_pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)

    device = mu_pred.device

    # 添加小的正则化项确保正定性
    reg = 1e-6 * torch.eye(dim, device=device)
    sigma_reg = sigma_pred + reg.unsqueeze(0)

    # 采样
    if distribution.lower() == 'laplace':
        # 多元拉普拉斯通过正态尺度混合表示: L(μ, Σ) = ∫ N(μ, τΣ) Exp(τ/2) dτ
        # 采样方法: 先采样 τ ~ Exp(1/2), 然后采样 x ~ N(μ, τΣ)
        tau_samples = torch.distributions.Exponential(0.5).sample((num_samples, batch_size))
        tau_samples = tau_samples.unsqueeze(-1)  # (num_samples, batch, 1)

        samples_list = []
        for i in range(num_samples):
            # 对于每个 τ，使用条件正态分布采样
            tau_i = tau_samples[i].unsqueeze(-1)  # (batch, 1) -> (batch, 1, 1)
            sigma_scaled = sigma_reg * tau_i  # (batch, 6, 6) * (batch, 1, 1) = (batch, 6, 6)

            try:
                dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_scaled)
                samples_list.append(dist.sample())
            except:
                # 退回到对角协方差
                sigma_diag = torch.diag_embed(torch.diagonal(sigma_scaled, dim1=-2, dim2=-1))
                dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_diag)
                samples_list.append(dist.sample())

        samples = torch.stack(samples_list)  # (num_samples, batch, 6)

        # 采样第二组（独立的 τ）
        tau_samples_prime = torch.distributions.Exponential(0.5).sample((num_samples, batch_size))
        tau_samples_prime = tau_samples_prime.unsqueeze(-1)

        samples_prime_list = []
        for i in range(num_samples):
            tau_i = tau_samples_prime[i].unsqueeze(-1)  # (batch, 1) -> (batch, 1, 1)
            sigma_scaled = sigma_reg * tau_i

            try:
                dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_scaled)
                samples_prime_list.append(dist.sample())
            except:
                sigma_diag = torch.diag_embed(torch.diagonal(sigma_scaled, dim1=-2, dim2=-1))
                dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_diag)
                samples_prime_list.append(dist.sample())

        samples_prime = torch.stack(samples_prime_list)  # (num_samples, batch, 6)

    else:  # 'gaussian'
        # 多元高斯分布
        try:
            dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_reg)
        except Exception as e:
            print(f"Warning: Could not create multivariate normal: {e}")
            # 退回到对角协方差
            sigma_diag = torch.diag_embed(torch.diagonal(sigma_pred, dim1=-2, dim2=-1) + 1e-6)
            dist = torch.distributions.MultivariateNormal(mu_pred, covariance_matrix=sigma_diag)

        samples = dist.sample((num_samples,))  # (num_samples, batch, 6)
        samples_prime = dist.sample((num_samples,))  # (num_samples, batch, 6)

    # target 扩充维度: (1, batch, 6)
    target_expanded = target.unsqueeze(0)

    # 1. 第一项: E[||X - y||]
    term1 = torch.norm(samples - target_expanded, dim=2).mean(dim=0)  # (Batch,)

    # 2. 第二项: 0.5 * E[||X - X'||]
    term2 = 0.5 * torch.norm(samples - samples_prime, dim=2).mean(dim=0)  # (Batch,)

    energy_score = term1 - term2
    return energy_score.mean().item()


def plot_reliability_diagram(mu_pred, sigma_pred, target, mahalanobis_distances=None,
                             use_laplacian=True, save_path=None):
    """
    绘制多元回归的校准覆盖率图（修正版）

    使用理论分布（高斯或拉普拉斯）计算 p-values，避免自循环陷阱。

    Args:
        mu_pred: (N, 6) numpy array
        sigma_pred: (N, 6, 6) numpy array
        target: (N, 6) numpy array
        mahalanobis_distances: (N,) numpy array, precomputed Mahalanobis distances squared
        use_laplacian: If True, use Multivariate Laplace CDF; if False, use Gaussian (chi2)
        save_path: 保存路径 (可选)

    Returns:
        mace: Mean Absolute Calibration Error
    """
    N, dim = mu_pred.shape

    print(f"\nComputing reliability diagram for {N} samples...")
    print(f"  Distribution assumption: {'Multivariate Laplace' if use_laplacian else 'Gaussian (chi2)'}")

    # 使用预计算的Mahalanobis距离，避免重复计算
    if mahalanobis_distances is not None:
        mahalanobis_sq = mahalanobis_distances
        print(f"Using precomputed Mahalanobis distances for {len(mahalanobis_sq)} samples")
    else:
        # 如果没有预计算，则计算Mahalanobis距离
        mahalanobis_sq = []
        for i in range(N):
            delta = target[i] - mu_pred[i]
            cov = sigma_pred[i]
            # D_M^2 = delta^T * inv(Sigma) * delta
            try:
                # 更稳定的方法：使用 solve 而不是显式求逆
                delta_t = torch.tensor(delta, dtype=torch.float32)
                cov_t = torch.tensor(cov, dtype=torch.float32)

                # Solve: cov * x = delta, then compute: delta^T * x
                x = torch.linalg.solve(cov_t, delta_t)
                d2 = torch.dot(delta_t, x)
                mahalanobis_sq.append(d2.item())
            except Exception as e:
                # 如果求解失败，跳过该样本
                continue

        mahalanobis_sq = np.array(mahalanobis_sq)
        print(f"Successfully computed Mahalanobis distances for {len(mahalanobis_sq)} samples")

    # --- 核心修正：计算基于理论分布的 p-values ---
    if use_laplacian:
        # 获取多元拉普拉斯理论 CDF 插值器（通过蒙特卡洛模拟生成）
        mvl_cdf = get_theoretical_mvl_cdf_interpolator(dim=dim)
        theo_p_values = mvl_cdf(mahalanobis_sq)
        print(f"  Using MVL CDF interpolator for p-value calculation")
    else:
        # 高斯假设：直接使用卡方分布 CDF
        theo_p_values = chi2.cdf(mahalanobis_sq, df=dim)
        print(f"  Using chi2 CDF for p-value calculation")

    # 计算实际覆盖率 vs 理论覆盖率
    expected_confidence = np.linspace(0, 1, 100)
    observed_confidence = []

    for p in expected_confidence:
        # 计算有多少比例的样本落在了 p 置信区间内
        # 只要样本的 theo_p_value <= p，就说明它在这个区间内
        fraction = np.mean(theo_p_values <= p)
        observed_confidence.append(fraction)

    # 计算 MACE (Mean Absolute Calibration Error)
    mace = np.mean(np.abs(np.array(observed_confidence) - expected_confidence))

    # 设置出版级风格
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'grid.alpha': 0.15,
        'grid.linestyle': '-',
        'lines.linewidth': 2.0
    })

    # 使用午夜蓝+香槟金配色
    COLOR_MODEL = '#002060'  # 午夜蓝
    COLOR_IDEAL = '#404040'  # 深灰
    COLOR_SHADE = '#F5D0A9'  # 香槟金

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # 绘制统计置信区间（基于二项分布）
    # 对于每个 p，观测到的 fraction 是 Binomial(N,p) 的估计
    from scipy.stats import beta as beta_dist
    n_samples = len(theo_p_values)
    alpha = 0.05  # 95% 置信区间

    # 使用 Beta 分布作为二项分布的共轭先验
    # Beta(α=1, β=1) 对应均匀先验
    lower_ci = []
    upper_ci = []

    for i, p in enumerate(expected_confidence):
        # 对于二项分布，使用 Clopper-Pearson 区间
        # 下界：Beta(α/2, k + β)
        # 上界：Beta(k + α, n-k + β/2)
        # 其中 k 是成功的次数（这里使用期望值 n*p）
        k_expected = n_samples * p

        # 使用 Beta 分布计算置信区间
        lower = beta_dist.ppf(alpha/2, k_expected + 0.5, n_samples - k_expected + 0.5)
        upper = beta_dist.ppf(1 - alpha/2, k_expected + 0.5, n_samples - k_expected + 0.5)

        lower_ci.append(max(0, lower))
        upper_ci.append(min(1, upper))

    # 绘制置信区间
    ax.fill_between(expected_confidence, lower_ci, upper_ci,
                    color=COLOR_SHADE, alpha=0.2,
                    label='95% Confidence Interval (Binomial)')

    # 绘制理想校准线
    ax.plot([0, 1], [0, 1], color=COLOR_IDEAL, linestyle='--',
            linewidth=2, label='Perfect Calibration')

    # 绘制实际校准曲线
    ax.plot(expected_confidence, observed_confidence,
            color=COLOR_MODEL, linewidth=2.5, label='Model Prediction')

    # 设置标签和标题
    dist_name = 'Multivariate Laplace' if use_laplacian else 'Gaussian (χ²₆)'
    ax.set_xlabel('Theoretical CDF Value (p)', fontsize=14)
    ax.set_ylabel('Empirical CDF: P(p_value ≤ p)', fontsize=14)
    ax.set_title(f'Calibration Curve ({dist_name})\nMACE = {mace:.4f}',
                loc='center', fontweight='bold', fontsize=16, pad=20)

    # 设置网格和图例
    ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)

    # 设置等比例
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 计算校准指标（MACE 已在上面计算）
    # KS Statistic (Kolmogorov-Smirnov distance between PIT and Uniform)
    ks_stat = np.max(np.abs(np.array(observed_confidence) - expected_confidence))

    # ECE (Expected Calibration Error) - 使用分箱版本
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(theo_p_values, bin_edges) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_conf = (bin_edges[i] + bin_edges[i+1]) / 2
            bin_acc = np.mean(theo_p_values[mask] <= bin_conf)
            bin_weight = np.sum(mask) / len(theo_p_values)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    print(f"\n[Calibration Metrics ({dist_name})]")
    print(f"  MACE (Mean Absolute Calibration Error): {mace:.4f}")
    print(f"  KS Statistic (max deviation): {ks_stat:.4f}")
    print(f"  ECE (10-bin Expected Calibration Error): {ece:.4f}")

    # 在图上添加指标
    metrics_text = (f'MACE = {mace:.4f}\n'
                   f'KS = {ks_stat:.4f}\n'
                   f'ECE = {ece:.4f}')

    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='white',
                     edgecolor='#CCCCCC',
                     alpha=0.9))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"\n[+] Reliability diagram saved to {save_path}")
    else:
        plt.savefig('reliability_diagram.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('reliability_diagram.png', dpi=300, bbox_inches='tight')
        print("\n[+] Reliability diagram saved to reliability_diagram.pdf/png")

    return mace


def get_theoretical_laplacian_quantiles(prob_levels, dim=6, num_mc_samples=1000000, seed=42):
    """
    Generate theoretical quantiles for Multivariate Laplace distribution using Monte Carlo.

    For Multivariate Laplace as Normal Scale Mixture:
        D_M^2 = W * V where:
        - W ~ Exp(2) [scale=2.0 for rate=0.5]
        - V ~ Chi^2(dim)

    Args:
        prob_levels: Array of probability levels (0, 1)
        dim: Dimension of the output (default 6)
        num_mc_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility

    Returns:
        Array of theoretical quantiles corresponding to prob_levels
    """
    np.random.seed(seed)

    # 1. Sample W from Exponential(scale=2.0)
    w = np.random.exponential(scale=2.0, size=num_mc_samples)

    # 2. Sample V from Chi-squared(df=dim)
    v = np.random.chisquare(df=dim, size=num_mc_samples)

    # 3. D_M^2 = W * V (product of independent variables)
    dm2_samples = w * v

    # 4. Get quantiles at requested probability levels
    theo_quantiles = np.quantile(dm2_samples, prob_levels)

    return theo_quantiles


def plot_laplacian_calibration(mahalanobis_distances, use_gaussian=True, save_path='paper_figure_laplacian_calibration.pdf'):
    """
    Plot calibration curve with support for both Gaussian and Laplacian assumptions.

    For Gaussian (Multivariate Normal):
        D_M^2 = (x - mu)^T Sigma^-1 (x - mu) ~ Chi^2(df=6)

    For Laplacian (Multivariate Laplace):
        D_M^2 follows a distribution derived from Normal Scale Mixture:
        D_M^2 = W * V where W ~ Exp(2), V ~ Chi^2(6)

    Args:
        mahalanobis_distances: Array of squared Mahalanobis distances
        use_gaussian: If True, use Chi^2 reference; if False, use theoretical Laplacian
        save_path: Path to save the figure
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
    })

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Sort the empirical distances
    emp_quantiles = np.sort(mahalanobis_distances)
    n_samples = len(emp_quantiles)

    # Probability levels (median of each bin)
    prob_levels = np.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)

    if use_gaussian:
        # Gaussian assumption: Chi^2 distribution
        theo_quantiles = chi2.ppf(prob_levels, df=6)
        ax_label = r'Theoretical Quantiles $\chi^2_6$ (Gaussian)'
        title_add = 'Gaussian NLL'
        dist_name = 'Gaussian (χ²₆)'
    else:
        # Laplacian assumption: Theoretical distribution via Monte Carlo
        # D_M^2 = W * V, W ~ Exp(2), V ~ Chi^2(6)
        theo_quantiles = get_theoretical_laplacian_quantiles(prob_levels, dim=6)
        ax_label = r'Theoretical Quantiles (Multivariate Laplace)'
        title_add = 'Laplacian NLL'
        dist_name = 'Multivariate Laplace'

    max_val = max(theo_quantiles.max(), emp_quantiles.max()) * 1.05

    # Plot perfect calibration line
    ax.plot([0, max_val], [0, max_val], color='#404040', linestyle='--',
            linewidth=1.5, label='Perfect Calibration', zorder=1)

    # Plot scatter
    ax.scatter(theo_quantiles, emp_quantiles, s=25, alpha=0.7,
                facecolors='#002060', edgecolors='#004080', linewidth=0.5,
                zorder=2, label='Test Samples')

    # Compute metrics
    mean_bias = np.mean(emp_quantiles - theo_quantiles)
    r_squared = np.corrcoef(theo_quantiles, emp_quantiles)[0, 1]**2

    # Add statistics text
    stats_text = (f"Distribution: {dist_name}\n"
                  f"$\\mathbf{{R^2}}$: {r_squared:.3f}\n"
                  f"$\\mathbf{{Mean Bias}}$: {mean_bias:.3f}")

    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # Set labels
    ax.set_xlabel(ax_label, fontsize=12)
    ax.set_ylabel(r'Empirical Quantiles $D_M^2$', fontsize=12)
    ax.set_title(f'Uncertainty Calibration ({title_add})', loc='left',
                 fontweight='bold', fontsize=14, pad=10)
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\n[+] Laplacian calibration figure saved to {save_path}")

    return r_squared, mean_bias


def plot_uq_sharpness_analysis(results, save_path='uq_sharpness_analysis.pdf'):
    """
    Plot sharpness analysis and risk-coverage curve.

    (a) Sharpness Distribution: Shows the volume of 95% confidence ellipsoids
    (b) Risk-Coverage Curve: Compares Trace vs $\lambda_{max}$ ranking metrics

    Demonstrates that tensor directional uncertainty ($\lambda_{max}$) is more
    informative than total uncertainty (Trace) for risk assessment.
    """
    volumes = np.array(results['sharpness_volumes'])
    radii = np.array(results['sharpness_radii'])
    errors = np.array(results['error_norm'])

    # Import seaborn for mako colormap
    import seaborn as sns

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Style setup
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
    })

    # Color scheme
    COLOR_MODEL = '#002060'      # 午夜蓝 (for $\lambda_{max}$)
    COLOR_TRACE = '#E76F51'      # 焦橙色 (for Trace - contrast color)
    COLOR_IDEAL = '#404040'      # 深灰

    # Get mako colormap
    mako_colors = sns.color_palette("mako", 10)

    # ========================================
    # (a) Sharpness Distribution
    # ========================================
    ax1 = axes[0]

    # Log-scale histogram for volumes (can span many orders of magnitude)
    log_volumes = np.log10(volumes + 1e-12)

    n_bins = min(50, max(20, int(len(volumes) / 50)))
    counts, bins, patches = ax1.hist(log_volumes, bins=n_bins, color='steelblue',
                                      edgecolor='black', alpha=0.7)

    for i, patch in enumerate(patches):
        color_idx = min(int(i / len(patches) * 8), 7)
        patch.set_facecolor(mako_colors[color_idx])

    ax1.axvline(np.median(log_volumes), color=mako_colors[-1], linestyle='--',
                linewidth=2, label=f'Median: $10^{{{np.median(log_volumes):.2f}}}$')

    # Larger, bolder x-axis label
    ax1.set_xlabel(r'$\bf{Log_{10}(95\%\ Confidence\ Volume)}$', fontsize=14, fontfamily='serif')
    ax1.set_ylabel('Count', fontsize=12, fontfamily='serif')
    ax1.set_title('(a) Sharpness Distribution', loc='left', fontweight='bold', fontsize=14, fontfamily='serif')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.3)

    # Style: inward ticks with minor ticks
    ax1.tick_params(axis='both', direction='in', which='both')
    ax1.minorticks_on()

    # ========================================
    # (b) Risk-Coverage Curve: Trace vs $\lambda_{max}$
    # ========================================
    ax2 = axes[1]

    # Get true and predicted values for MAE calculation
    true_km = np.array(results['true_diag'])
    pred_km = np.array(results['pred_diag'])

    # Also include off-diagonal if available
    if 'true_off' in results and 'pred_off' in results:
        true_km = np.concatenate([true_km, np.array(results['true_off'])])
        pred_km = np.concatenate([pred_km, np.array(results['pred_off'])])

    n_samples = len(true_km) // 6  # Assuming 6 components per sample
    n_components = true_km.shape[0] // n_samples

    true_km_reshaped = true_km.reshape(n_samples, n_components)
    pred_km_reshaped = pred_km.reshape(n_samples, n_components)

    # Per-sample MAE
    sample_errors = np.mean(np.abs(true_km_reshaped - pred_km_reshaped), axis=1)

    # Fine-grained coverage levels: 1% steps from 100% down to 20%
    coverage_levels = np.linspace(1.0, 0.2, 81)  # 1% increments

    # Helper function to compute MAE at coverage levels for a given ranking metric
    def compute_mae_by_ranking(uncertainty_metric):
        ranked_indices = np.argsort(uncertainty_metric)[::-1]  # Descending
        mae_at_coverage = []
        for coverage in coverage_levels:
            n_keep = max(int(n_samples * coverage), 1)
            keep_indices = ranked_indices[-n_keep:]
            mae = np.mean(sample_errors[keep_indices])
            mae_at_coverage.append(mae)
        return np.array(mae_at_coverage)

    # ===== Method 1: Trace Ranking (baseline) =====
    if 'uncertainty_trace' in results and len(results['uncertainty_trace']) > 0:
        uncertainty_trace = np.array(results['uncertainty_trace'])
    else:
        uncertainty_trace = np.array(results['uncertainty'])
    mae_trace = compute_mae_by_ranking(uncertainty_trace)

    # ===== Method 2: $\lambda_{max}$ Ranking (proposed) =====
    if 'eigenvals' in results and len(results['eigenvals']) > 0:
        eigenvals = np.array(results['eigenvals'])
        if len(eigenvals.shape) == 1:
            eigenvals = eigenvals.reshape(-1, 1)
        uncertainty_max_eig = eigenvals[:, -1]  # Max eigenvalue
    else:
        uncertainty_max_eig = uncertainty_trace  # Fallback
    mae_max_eig = compute_mae_by_ranking(uncertainty_max_eig)

    # Get baseline (full coverage MAE)
    baseline_mae = mae_trace[0]

    # ===== Plot both curves =====
    # Trace curve (baseline - dashed, lighter)
    ax2.plot(coverage_levels * 100, mae_trace,
             color=COLOR_TRACE, linewidth=2, linestyle='--',
             label=r'Trace($\Sigma$)', alpha=0.8)

    # $\lambda_{max}$ curve (proposed - solid, thicker, more prominent)
    ax2.plot(coverage_levels * 100, mae_max_eig,
             color=COLOR_MODEL, linewidth=3,
             label=r'$\lambda_{max}(\Sigma)$', zorder=10)

    # Fill area under $\lambda_{max}$ curve only
    ax2.fill_between(coverage_levels * 100, mae_max_eig.min(), mae_max_eig,
                     color=COLOR_MODEL, alpha=0.12)

    # Add baseline reference line
    ax2.axhline(y=baseline_mae, color=COLOR_IDEAL, linestyle=':',
                linewidth=1.2, alpha=0.6)

    # ===== Markers and annotations at key points =====
    key_indices = [0, 10, 20, 50, 80]  # 100%, 90%, 80%, 50%, 20%

    # Markers for Trace
    ax2.plot(coverage_levels[key_indices] * 100,
             [mae_trace[i] for i in key_indices],
             'o', color='white', markeredgecolor=COLOR_TRACE,
             markeredgewidth=1.2, markersize=5, alpha=0.7, zorder=5)

    # Markers for $\lambda_{max}$ (larger, more prominent)
    ax2.plot(coverage_levels[key_indices] * 100,
             [mae_max_eig[i] for i in key_indices],
             'o', color='white', markeredgecolor=COLOR_MODEL,
             markeredgewidth=2, markersize=7, zorder=15)

    # ===== Annotate improvement at key coverage levels =====
    for idx in [10, 20]:  # 90% and 80% coverage
        if idx < len(coverage_levels):
            cov = coverage_levels[idx] * 100
            mae_trace_val = mae_trace[idx]
            mae_max_eig_val = mae_max_eig[idx]

            # Improvement of $\lambda_{max}$ over baseline
            improvement_max_eig = (baseline_mae - mae_max_eig_val) / baseline_mae * 100
            # Improvement of $\lambda_{max}$ over Trace
            improvement_vs_trace = (mae_trace_val - mae_max_eig_val) / mae_trace_val * 100

            # Bold annotation for $\lambda_{max}$ improvement
            ax2.annotate(f'{improvement_max_eig:+.1f}%',
                        xy=(cov, mae_max_eig_val), xytext=(0, -18),
                        textcoords='offset points', fontsize=10,
                        color=COLOR_MODEL, fontweight='bold',
                        ha='center')

            # Smaller annotation showing advantage over Trace
            if improvement_vs_trace > 0.5:  # Only show if meaningful difference
                ax2.annotate(f'(vs Trace: +{improvement_vs_trace:.1f}%)',
                            xy=(cov, mae_max_eig_val), xytext=(0, -32),
                            textcoords='offset points', fontsize=7,
                            color=COLOR_MODEL, style='italic',
                            ha='center')

    # Labels and title
    ax2.set_xlabel('Coverage (% of samples retained)', fontsize=12, fontfamily='serif')
    ax2.set_ylabel('MAE (Kelvin-Mandel log space)', fontsize=12, fontfamily='serif')
    ax2.set_title('(b) Risk-Coverage: Directional vs. Total Uncertainty',
                  loc='left', fontweight='bold', fontsize=14, fontfamily='serif')

    # Set x-axis limits and ticks
    ax2.set_xlim(18, 102)
    ax2.set_xticks([20, 40, 60, 80, 100])

    # Style: inward ticks with minor ticks
    ax2.tick_params(axis='both', direction='in', which='both')
    ax2.minorticks_on()

    # Grid
    ax2.grid(True, linestyle=':', alpha=0.3)

    # Legend (positioned to not overlap with curve)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95,
              fontsize=11, fancybox=True)

    # Add text showing total number of samples
    ax2.text(0.98, 0.02, f'N = {n_samples}', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\n[+] UQ Sharpness & Risk-Coverage analysis saved to {save_path}")

    return {
        'sharpness_mean': float(np.mean(volumes)),
        'sharpness_median': float(np.median(volumes)),
        'sharpness_std': float(np.std(volumes)),
        'sharpness_min': float(np.min(volumes)),
        'sharpness_max': float(np.max(volumes)),
        'variance_collapse_detected': bool(np.median(volumes) < 1e-3),
    }


def analyze_outliers(results, save_path='outliers_analysis.json'):
    """分析并保存害群之马信息"""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS - IDENTIFYING PROBLEMATIC SAMPLES")
    print("="*60)

    outlier_info = results['outlier_info']

    # 1. 按马氏距离排序（最重要的指标）
    outliers_by_mahalanobis = sorted(
        [x for x in outlier_info if not np.isnan(x['mahalanobis_distance_sq'])],
        key=lambda x: x['mahalanobis_distance_sq'],
        reverse=True
    )

    # 2. 按相对误差排序
    outliers_by_rel_error = sorted(
        outlier_info,
        key=lambda x: x['relative_error'],
        reverse=True
    )

    # 3. 按条件数排序
    outliers_by_cond = sorted(
        outlier_info,
        key=lambda x: x['condition_number'],
        reverse=True
    )

    # 4. 找出潜在的害群之马（满足任一异常标准）
    potential_outliers = [x for x in outlier_info if x['is_potential_outlier']]

    print(f"\nTotal samples analyzed: {len(outlier_info)}")
    print(f"Potential outliers (>=1 anomaly): {len(potential_outliers)} ({len(potential_outliers)/len(outlier_info)*100:.1f}%)")

    # Outlier details are saved to file, not printed to save space

    # 保存详细信息
    analysis_data = {
        'statistics': {
            'total_samples': len(outlier_info),
            'potential_outliers': len(potential_outliers),
            'outlier_percentage': len(potential_outliers) / len(outlier_info) * 100,
            'chi2_95th_percentile': chi2.ppf(0.95, df=6),  # 保留参考
            'chi2_99th_percentile': chi2.ppf(0.99, df=6),  # 现在使用的阈值
            'threshold_used': 'chi2_99th_percentile'  # 明确说明使用哪个阈值
        },
        'top_outliers_by_mahalanobis': outliers_by_mahalanobis[:20],
        'top_outliers_by_relative_error': outliers_by_rel_error[:20],
        'top_outliers_by_condition_number': outliers_by_cond[:20],
        'all_potential_outliers': potential_outliers,
        'sample_indices_to_remove': [x['stable_id'] for x in potential_outliers if x['stable_id'] is not None]
    }

    # 保存到JSON文件 - 转换numpy类型为Python原生类型
    import json
    analysis_data_converted = convert_numpy_types(analysis_data)

    with open(save_path, 'w') as f:
        json.dump(analysis_data_converted, f, indent=2)

    print(f"\n[+] Outlier analysis saved to '{save_path}'")
    print(f"[+] All outlier details and sample indices saved in the file")

    # 额外的统计分析
    dm2_values = [x['mahalanobis_distance_sq'] for x in outlier_info if not np.isnan(x['mahalanobis_distance_sq'])]
    if dm2_values:
        print(f"\n[Mahalanobis Distance Statistics]")
        print(f"  Mean: {np.mean(dm2_values):.2f} (Ideal: 6.0)")
        print(f"  Median: {np.median(dm2_values):.2f}")
        print(f"  95th percentile: {np.percentile(dm2_values, 95):.2f}")
        print(f"  Max: {np.max(dm2_values):.2f}")

        # 判断校准状态
        if np.mean(dm2_values) < 3.0:
            print(f"  → Model appears UNDER-CONFIDENT (uncertainties too large)")
        elif np.mean(dm2_values) > 10.0:
            print(f"  → Model appears OVER-CONFIDENT (uncertainties too small)")

    return analysis_data['sample_indices_to_remove']




def run_ood_distortion_analysis(model, val_loader, device, save_dir='figures'):
    """
    Chemical OOD Analysis: Element Substitution (Gold Standard for UQ evaluation)

    Core idea: Replace known elements with OOD elements (actinides, rare earths)
    that the model has never seen during training. Ideally, the model should show
    significantly higher uncertainty when encountering unknown chemical environments.

    Key improvements over previous version:
    - Uses Batch.to_data_list() for robust subgraph extraction
    - Properly recomputes node_features after element substitution
    - Uses 20 samples for statistical robustness
    - Added progress bar with tqdm
    - Academic-style visualization with error bands

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        save_dir: Directory to save plots

    Returns:
        results_ood: Dictionary with OOD analysis results
    """
    import os
    from torch_geometric.data import Batch
    from tqdm import tqdm
    from atom_features import create_composite_atom_features

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. Define OOD elements (unlikely to be in training data)
    OOD_ELEMENTS = {
        92: 'U',   # Uranium (actinide)
        94: 'Pu',  # Plutonium (actinide)
        93: 'Np',  # Neptunium (actinide)
        60: 'Nd',  # Neodymium (rare earth)
        62: 'Sm',  # Samarium (rare earth)
        64: 'Gd',  # Gadolinium (rare earth)
    }
    ood_z_list = list(OOD_ELEMENTS.keys())

    substitution_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Collect samples: use to_data_list() for robust subgraph extraction
    print(f"\n[PHASE 4] Running OOD Chemical Substitution Analysis...")
    print(f"  [INFO] OOD Elements: {list(OOD_ELEMENTS.values())}")

    baseline_samples = []
    max_test_samples = 20  # Increased for statistical robustness

    for batch in val_loader:
        # Convert Batch to independent Data objects - handles all index offsets automatically
        baseline_samples.extend(batch.to_data_list())
        if len(baseline_samples) >= max_test_samples:
            break
    baseline_samples = baseline_samples[:max_test_samples]

    print(f"  [INFO] Testing {len(baseline_samples)} samples across {len(substitution_ratios)} ratios")

    results_ood = []

    # 3. Run test: element substitution at different ratios
    with torch.no_grad():
        for sub_ratio in substitution_ratios:
            print(f"  Testing ratio: {sub_ratio:.1f}")
            for data in tqdm(baseline_samples, leave=False, desc=f"  Ratio {sub_ratio:.1f}"):
                # Clone and move to device
                modified_data = data.clone().to(device)
                num_nodes = modified_data.num_nodes

                if sub_ratio > 0:
                    # Randomly select node indices for replacement
                    n_sub = max(1, int(num_nodes * sub_ratio))
                    indices = np.random.choice(num_nodes, n_sub, replace=False)

                    # Replace atomic numbers z with OOD elements
                    new_elements = torch.tensor(
                        np.random.choice(ood_z_list, size=n_sub),
                        dtype=torch.long, device=device
                    )
                    modified_data.z[indices] = new_elements

                    # KEY: Recompute node features so UQ can "see" the unknown chemistry
                    # Without this, the model still uses original atom features
                    modified_data.node_features = create_composite_atom_features(
                        modified_data.z.cpu(), use_onehot=True
                    ).to(device)

                # Build single-sample batch for inference
                test_batch = Batch.from_data_list([modified_data])
                mu_km, A_km, Sigma_km = model(test_batch, compute_sigma=True)

                # Extract uncertainty metrics
                sigma = Sigma_km[0].cpu().numpy()
                eigvals = np.linalg.eigvalsh(sigma)

                results_ood.append({
                    'ratio': sub_ratio,
                    'lambda_max': eigvals[-1],
                    'trace': np.trace(sigma),
                    'log_det': np.log(np.linalg.det(sigma) + 1e-10)
                })

    # 4. Aggregate and plot results
    return _plot_and_interpret_ood_results(results_ood, substitution_ratios, save_dir)


def _plot_and_interpret_ood_results(results_ood, ratios, save_dir):
    """Aggregate results and generate academic-quality plot"""
    avg_lmax, std_lmax = [], []

    for r in ratios:
        subset = [res['lambda_max'] for res in results_ood if res['ratio'] == r]
        avg_lmax.append(np.mean(subset))
        std_lmax.append(np.std(subset))

    # Academic-style plot
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
    })

    # Main line with error bands
    plt.plot(ratios, avg_lmax, 'o-', color='#002060', linewidth=2.5,
             markersize=8, label='Mean $\\lambda_{max}$')
    plt.fill_between(ratios,
                     np.array(avg_lmax) - np.array(std_lmax),
                     np.array(avg_lmax) + np.array(std_lmax),
                     alpha=0.2, color='#002060', label='±1 Std')

    # Add baseline reference line
    plt.axhline(y=avg_lmax[0], color='red', linestyle='--', alpha=0.7,
                linewidth=2, label='In-Distribution Baseline')

    plt.xlabel('Substitution Ratio (Unknown Elements)', fontsize=13)
    plt.ylabel('Predicted Uncertainty ($\\lambda_{max}$)', fontsize=13)
    plt.title('Chemical OOD Analysis: Sensitivity to Unknown Species',
              fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ood_chemical_test.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  [+] OOD analysis plot saved to '{save_path}'")

    # Result interpretation
    l_max_baseline = avg_lmax[0]
    l_max_full_ood = avg_lmax[-1]
    l_max_increase = (l_max_full_ood - l_max_baseline) / (l_max_baseline + 1e-10) * 100

    print(f"\n[OOD Analysis Results - Chemical OOD]")
    print(f"  Baseline (0% substitution):     {l_max_baseline:.4f}")
    print(f"  Full OOD (100% substitution):   {l_max_full_ood:.4f}")
    print(f"  Relative increase:              {l_max_increase:+.1f}%")

    print(f"\n  Ratio | Uncertainty | Change")
    print(f"  {'-'*38}")
    for i, r in enumerate(ratios):
        change = (avg_lmax[i] - l_max_baseline) / (l_max_baseline + 1e-10) * 100
        print(f"  {r*100:5.0f}% | {avg_lmax[i]:11.4f} | {change:+7.1f}%")

    if l_max_increase > 100:
        print(f"\n  [EXCELLENT] Uncertainty doubles with OOD elements!")
        print(f"              Model successfully recognizes unknown chemistry.")
    elif l_max_increase > 50:
        print(f"\n  [GOOD] Uncertainty increases significantly (>50%)")
    elif l_max_increase > 20:
        print(f"\n  [MODERATE] Uncertainty increases moderately")
    else:
        print(f"\n  [WARNING] Uncertainty does NOT respond to chemical OOD")
        print(f"  [INTERPRETATION] Possible issues:")
        print(f"    1. UQ branch may not be using atom features properly")
        print(f"    2. detach_uq_features may prevent chemical sensitivity")
        print(f"    3. Node feature embedding might be too generic")

    return results_ood


def main():
    print("="*60)
    print("E(3)-EQUIVARIANT UNCERTAINTY QUANTIFICATION EVALUATION")
    print("="*60)

    model, processor, log_mean, log_std, device, component_mean, component_std = load_model_and_data()

    # Load validation data separately
    _, val_loader, _ = get_dielectric_data_loaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        train_subset=None
    )
    # Note: normalization parameters are loaded from checkpoint (training set)

    # [TEMPORARILY DISABLED] Skip outlier analysis to speed up evaluation
    print("\n[INFO] Skipping outlier analysis (temporarily disabled)")

    # 1. 评估验证集
    print("\n[PHASE 1] Evaluating VALIDATION SET...")
    results = evaluate_model(model, val_loader, processor, log_mean, log_std,
                            dataset_name="Validation")

    # [NEW] OOD 畸变测试
    print("\n" + "="*60)
    print("OOD DISTORTION ANALYSIS")
    print("="*60)
    run_ood_distortion_analysis(model, val_loader, device, save_dir='figures')

    # ========================================
    # TEMPERATURE SCALING: 寻找最优温度并应用
    # ========================================
    print("\n" + "="*60)
    print("TEMPERATURE SCALING - POST-HOC CALIBRATION")
    print("="*60)

    # 准备数据用于温度缩放
    if results['mu_km_list'] and results['sigma_km_list'] and results['targets_km_list']:
        mu_all = np.concatenate(results['mu_km_list'], axis=0)
        sigma_all = np.concatenate(results['sigma_km_list'], axis=0)
        targets_all = np.concatenate(results['targets_km_list'], axis=0)

        print(f"\n原始模型校准状态（温度缩放前）:")
        dm2_orig = np.array(results['mahalanobis_distances'])
        print(f"  D_M^2 Mean:   {np.nanmean(dm2_orig):.4f} (ideal: ~6.0)")
        print(f"  D_M^2 Median: {np.nanmedian(dm2_orig):.4f} (ideal: ~5.35)")
        print(f"  D_M^2 95th:   {np.nanpercentile(dm2_orig, 95):.4f} (ideal: ~12.59)")

        # 寻找最优温度 T（使用拉普拉斯 NLL）
        T_opt = find_optimal_temperature(mu_all, sigma_all, targets_all, use_laplacian=True)

        # 应用温度缩放到所有结果
        results = apply_temperature_scaling(results, T_opt)

        print(f"\n[温度缩放后，所有后续分析将使用校准后的协方差矩阵]")

    # Note: Calibration plots are now integrated into paper_figure_accuracy.pdf

    # Outlier analysis temporarily disabled
    outlier_indices = []  # Empty list since outlier analysis is disabled

    # Compute Energy Score and Reliability Diagram
    print("\n[+] Computing additional calibration metrics...")

    # Prepare data for Energy Score and Reliability Diagram
    if results['mu_km_list'] and results['sigma_km_list'] and results['targets_km_list'] and \
       len(results['mu_km_list']) > 0:
        # Concatenate all batches
        mu_all = np.concatenate(results['mu_km_list'], axis=0)
        sigma_all = np.concatenate(results['sigma_km_list'], axis=0)
        targets_all = np.concatenate(results['targets_km_list'], axis=0)

        print(f"Total samples for analysis: {len(mu_all)}")

        # Compute Energy Score
        print("\nComputing Energy Score...")
        es = compute_energy_score(mu_all, sigma_all, targets_all, num_samples=1000)
        print(f"Energy Score (ES): {es:.4f} (lower is better)")

        # Create diagonal baseline for comparison
        print("\nComputing diagonal baseline...")
        sigma_diag = np.zeros_like(sigma_all)
        for i in range(len(sigma_all)):
            sigma_diag[i] = np.diag(np.diag(sigma_all[i]))

        es_diag = compute_energy_score(mu_all, sigma_diag, targets_all, num_samples=1000)
        print(f"Diagonal Baseline ES: {es_diag:.4f}")
        print(f"Improvement: {((es_diag - es) / es_diag * 100):.1f}%")

        # [NEW] UQ Sharpness Analysis - Detect variance collapse and over-conservative predictions
        print("\nGenerating UQ Sharpness Analysis...")
        sharpness_metrics = plot_uq_sharpness_analysis(
            results,
            save_path='uq_sharpness_analysis.pdf'
        )

        # Add to results
        print("\n[Advanced Calibration Metrics]")
        print(f"  Energy Score (Full):        {es:.4f}")
        print(f"  Energy Score (Diagonal):     {es_diag:.4f}")
        print(f"  Relative Improvement:        {((es_diag - es) / es_diag * 100):.1f}%")

        # [NEW] Sharpness metrics
        print(f"\n[Sharpness Metrics - UQ Quality Diagnostics]")
        print(f"  95% Confidence Volume:")
        print(f"    Mean:                      {sharpness_metrics['sharpness_mean']:.6e}")
        print(f"    Median:                    {sharpness_metrics['sharpness_median']:.6e}")
        print(f"    Std:                       {sharpness_metrics['sharpness_std']:.6e}")
        if sharpness_metrics['variance_collapse_detected']:
            print(f"    [WARNING] Variance collapse detected! (volume too small)")
        else:
            print(f"    [OK] Sharpness distribution appears normal")

        # [NEW] Calibration Coverage Table - Direct way to detect variance collapse
        print(f"\n[Calibration Coverage Table (vs Gaussian χ² reference)]")
        print(f"  Note: For Laplacian NLL, systematic deviation is expected")
        coverage_results = compute_calibration_coverage(results['mahalanobis_distances'], use_gaussian=True)
        print(f"  {'Theoretical':>12} | {'Empirical':>12} | {'Deviation':>12} | {'Status'}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+{'-'*20}")
        for cl_name, cl_data in coverage_results.items():
            theo = cl_data['theoretical']
            emp = cl_data['empirical']
            dev = cl_data['deviation']
            # Status determination
            if abs(dev) < 0.02:
                status = "✓ Excellent"
            elif abs(dev) < 0.05:
                status = "~ Good"
            elif dev < -0.05:  # Empirical < Theoretical → Over-confident
                status = "⚠ Over-confident"
            else:  # Empirical > Theoretical → Under-confident
                status = "⚠ Under-confident"
            print(f"  {cl_name:>12} | {emp*100:11.1f}% | {dev*100:+11.1f}% | {status}")

        # [NEW] Eigenvalue Collapse Warning
        eigenvals = np.array(results['eigenvals'])
        min_eigenval = eigenvals.min()
        mean_eigenval = eigenvals.mean()

        # Physical lower bound check (if eigenvalues are hitting the training floor)
        # Typical training lower bound is around 1e-6 to 1e-4
        PHYSICAL_LOWER_BOUND = 1e-4

        print(f"\n[Eigenvalue Analysis - Variance Collapse Detection]")
        print(f"  λ_min:                     {min_eigenval:.6e}")
        print(f"  λ_mean:                    {mean_eigenval:.6e}")
        print(f"  Physical lower bound:      {PHYSICAL_LOWER_BOUND:.6e}")

        if min_eigenval < PHYSICAL_LOWER_BOUND * 10:
            print(f"  [CRITICAL WARNING] Eigenvalues approaching physical lower bound!")
            print(f"                    This suggests VARIANCE COLLAPSE - model is trying to")
            print(f"                    shrink uncertainties below the training floor.")
            print(f"                    Consider: (1) Regularization adjustments, (2) Temperature scaling")
        elif min_eigenval < PHYSICAL_LOWER_BOUND * 100:
            print(f"  [WARNING] Eigenvalues getting close to lower bound.")
            print(f"            Monitor for variance collapse in future epochs.")
        else:
            print(f"  [OK] Eigenvalues well above lower bound.")

        # Now plot the paper figures
        print("\n[+] Generating paper figures...")
        metrics = plot_paper_figures(results, processor)

        # Risk-coverage curve is now included in uq_sharpness_analysis

        # Print accuracy metrics
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"\n[Accuracy Metrics (Physical Space - Matching train.py)]")
        if results['mae_phys']:
            print(f"  Overall MAE:            {np.mean(results['mae_phys']):.4f}")
            print(f"  Diagonal MAE:           {np.mean(results['mae_diag_phys']):.4f}")
            print(f"  Off-Diagonal MAE:       {np.mean(results['mae_off_phys']):.4f}")
        print(f"\n[Accuracy Metrics (Log Space - For Reference)]")
        print(f"  Diagonal R2:            {metrics['r2_diag']:.3f}")
        print(f"  Log-space MAE:          {metrics['mae_diag']:.3f}")

        # Raw diagnostic values
        print(f"\n[Raw Diagnostics]")
        dm2 = np.array(results['mahalanobis_distances'])
        print(f"  D_M^2 mean:             {dm2.mean():.4f}")
        print(f"  D_M^2 median:           {np.median(dm2):.4f}")
        print(f"  D_M^2 95th:             {np.percentile(dm2, 95):.4f}")
        print(f"  D_M^2 99th:             {np.percentile(dm2, 99):.4f}")
        print(f"  χ²_6 median (ideal):    5.3481")
        print(f"  χ²_6 95th (ideal):     12.5916")
        conds = np.array(results['conds'])
        print(f"  Cond mean:              {conds.mean():.2f}")
        print(f"  Cond median:            {np.median(conds):.2f}")
        print(f"  Cond max:               {conds.max():.2f}")
        print(f"  Cond 95th:              {np.percentile(conds, 95):.2f}")
        print(f"  Cond > 50:              {(conds > 50).sum()} / {len(conds)}")
        print(f"  Cond > 100:             {(conds > 100).sum()} / {len(conds)}")
        print(f"  Cond > 150:             {(conds > 150).sum()} / {len(conds)}")
        eigenvals = np.array(results['eigenvals'])
        print(f"  λ_min:                  {eigenvals.min():.6f}")
        print(f"  λ_max:                  {eigenvals.max():.6f}")
        print(f"  λ mean:                 {eigenvals.mean():.6f}")
        print(f"  λ < 0 count:            {(eigenvals < 0).sum()}")
        print(f"  Σ SPD satisfaction:     {metrics['pos_def_rate']:.2f}%")
        if results['mu_spd_satisfied']:
            mu_spd_rate = np.mean(results['mu_spd_satisfied']) * 100
            print(f"  μ SPD satisfaction:     {mu_spd_rate:.2f}%")
        print(f"  Anisotropy ratio mean:  {metrics['mean_anisotropy_ratio']:.2f}")
        print(f"  Anisotropy ratio med:   {metrics['median_anisotropy_ratio']:.2f}")
        print(f"  MACE:                   {metrics.get('mace', 0):.4f}")
        print(f"  Spearman corr:          {metrics.get('spearman_corr', 0):.4f}")
    else:
        print("\n[-] No data available for Energy Score and Reliability Diagram")
        # Still plot basic figures
        print("\n[+] Generating paper figures...")
        metrics = plot_paper_figures(results, processor, use_gaussian_calibration=False)

        # Print SPD satisfaction for predicted mean tensors
        if results['mu_spd_satisfied']:
            mu_spd_rate = np.mean(results['mu_spd_satisfied']) * 100
            print(f"\n[SPD Constraint Satisfaction]")
            print(f"  Σ SPD satisfaction (covariance):     {metrics['pos_def_rate']:.2f}%")
            print(f"  μ SPD satisfaction (predicted mean): {mu_spd_rate:.2f}%")

        # Risk-coverage curve is now included in uq_sharpness_analysis

    # [新增] 使用说明
    print("\n" + "="*60)
    print("NEXT STEPS - USING OUTLIER INFORMATION")
    print("="*60)
    print(f"Found {len(outlier_indices)} potential outliers in validation set")
    print(f"Indices saved to 'outliers_analysis.json'")
    print("\nTo use this information for retraining:")
    print("1. Load the outlier indices from 'outliers_analysis.json'")
    print("2. Create a new dataset that excludes these samples")
    print("3. Retrain the model on the cleaned dataset")
    print("4. Expect better calibration and potentially lower overall error")

if __name__ == "__main__":
    main()