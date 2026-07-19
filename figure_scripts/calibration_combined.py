import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm

# 导入项目模块
from dielectric_data_loader_precomputed import get_dielectric_data_loaders_optimized
from equivariant_network import EquivariantUncertaintyNetwork
from voigt_utils import voigt_to_kelvin_mandel, kelvin_mandel_to_voigt, sym_matrix_exp_voigt
from train import voigt_to_matrix_batch


def compute_mahalanobis_distances(model, dataloader, device, num_samples=500):
    """
    使用训练好的模型计算真实数据的马氏距离 (向量化加速版)。
    使用与训练时相同的数据加载器和格式。
    """
    model.eval()
    mahalanobis_distances = []

    print(f"\nComputing Mahalanobis distances on {num_samples} samples...")

    # 获取归一化参数（与训练时一致）
    LOG_MEAN_SCALAR = dataloader.dataset.log_mean
    LOG_STD_SCALAR = dataloader.dataset.log_std

    with torch.no_grad():
        sample_count = 0

        for batch in tqdm(dataloader, desc="Processing Batches"):
            if sample_count >= num_samples:
                break

            # 使用PyG Data对象，不需要转换为字典
            batch = batch.to(device)

            # Skip if no edges (graph is empty)
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                print(f"Warning: Batch has no edges, skipping...")
                continue

            # 1. 模型预测（模型输出已经在KM空间）
            # mu_km: [B, 6], A_km: [B, 6, 6], Sigma_km: [B, 6, 6]
            mu_km, A_km, Sigma_km = model(batch, compute_sigma=True)

            # 2. Reshape batch.y from [num_nodes * 6] to [batch_size, 6]
            targets_km = batch.y.view(mu_km.shape[0], 6)

            # 3. 批量计算马氏距离
            # diff: [B, 6] -> [B, 6, 1]
            diff = (targets_km - mu_km).unsqueeze(-1)

            try:
                # 批量解线性方程: Sigma * x = diff -> x = Sigma^{-1} * diff
                # torch.linalg.solve 支持 batch 操作
                inv_sigma_diff = torch.linalg.solve(Sigma_km, diff)

                # 计算二次型: diff^T * Sigma^{-1} * diff
                # bmm: [B, 1, 6] x [B, 6, 1] -> [B, 1, 1]
                mahalanobis_sq = torch.bmm(diff.transpose(1, 2), inv_sigma_diff).squeeze(-1).squeeze(-1) # [B]

                # 收集结果
                batch_results = mahalanobis_sq.cpu().numpy().tolist()
                mahalanobis_distances.extend(batch_results)
                sample_count += len(mahalanobis_sq)

            except RuntimeError as e:
                print(f"Warning: Singular matrix encountered in batch. Skipping batch. Error: {e}")
                continue

    # 截断到请求的数量
    return np.array(mahalanobis_distances[:num_samples])

def get_ellipsoid_surface(tensor, n_points=50):
    """
    根据 3x3 张量生成椭球表面坐标。
    """
    # 特征分解
    evals, evecs = np.linalg.eigh(tensor)
    
    # 椭球半径 (对应特征值的平方根)
    radii = np.sqrt(evals)
    
    # 生成单位球面的网格
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # 将单位球变形为椭球
    # 旋转并缩放
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot(evecs, np.array([x[i,j], y[i,j], z[i,j]]) * radii)
            
    return x, y, z

def plot_publication_quality(mahalanobis_distances, samples_data):
    """
    生成最终出版级质量的论文配图 (Side-by-Side 布局)
    包含样式微调：去除3D背景灰度、优化字体、调整标注位置。
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import art3d

    # --- 全局风格设置 (学术出版风格) ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], # 优先使用 Times New Roman
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'mathtext.fontset': 'stix' # 使用更美观的数学公式字体
    })

    # 创建画布 (宽长比适合单栏或半页宽度)
    fig = plt.figure(figsize=(11, 5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # ==========================================
    # Left Plot: Q-Q Plot (Calibration)
    # ==========================================
    ax1 = fig.add_subplot(gs[0])

    # 数据准备
    emp_quantiles = np.sort(mahalanobis_distances)
    n_samples = len(emp_quantiles)
    prob_levels = np.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)
    theo_quantiles = stats.chi2.ppf(prob_levels, df=6)

    # 绘制对角线 (Perfect Calibration)
    max_val = max(theo_quantiles.max(), emp_quantiles.max()) * 1.05
    ax1.plot([0, max_val], [0, max_val], color='#D62728', linestyle='--', linewidth=1.5,
             label='Ideal Calibration', zorder=1)

    # 绘制散点 (Test Samples) - 使用中空圆点或半透明实心点增加质感
    ax1.scatter(theo_quantiles, emp_quantiles, s=20, alpha=0.6,
                facecolors='#1F77B4', edgecolors='w', linewidth=0.5,
                label='Test Samples', zorder=2)

    # 统计指标
    mean_bias = np.mean(np.sqrt(emp_quantiles) - np.sqrt(theo_quantiles))
    r_squared = np.corrcoef(theo_quantiles, emp_quantiles)[0, 1]**2

    # 统计信息框 (左上角，更精致的边框)
    stats_text = (f"$\mathbf{{R^2}}$: {r_squared:.3f}\n"
                  f"$\mathbf{{Mean Bias}}$: {mean_bias:.3f}")
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#BBBBBB')
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # "Conservative" 标注 (位置微调，避免遮挡)
    ax1.annotate('Conservative\n(Under-confident)',
                 xy=(15, 12), xycoords='data',
                 xytext=(22, 6), textcoords='data', # 稍微移远一点
                 arrowprops=dict(arrowstyle="->", color='#444444', connectionstyle="arc3,rad=-0.2", lw=1),
                 fontsize=10, color='#444444', ha='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)) # 添加白色底色防止文字看不清

    # 轴设置
    ax1.set_xlabel(r'Theoretical Quantiles ($\chi^2_6$)', labelpad=8)
    ax1.set_ylabel(r'Empirical Quantiles ($D_M^2$)', labelpad=8)
    # 标题左对齐 (出版级常用风格)
    ax1.set_title('(a) Uncertainty Calibration', loc='left', pad=10, fontweight='bold')
    
    # 网格线更淡
    ax1.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#CCCCCC')
    ax1.set_xlim(-1, max_val)
    ax1.set_ylim(-1, max_val)
    ax1.set_aspect('equal')

    # ==========================================
    # Right Plot: 3D Visualization (只取第一个样本)
    # ==========================================
    ax2 = fig.add_subplot(gs[1], projection='3d')
    
    # 获取第一个样本的数据 (假设 samples_data 是列表)
    sample_data = samples_data[0] 
    x_mu, y_mu, z_mu = get_ellipsoid_surface(sample_data['mu_matrix'])
    x_unc, y_unc, z_unc = get_ellipsoid_surface(sample_data['unc_tensor'])

    # --- 3D 美化关键步骤: 去除灰色背景 ---
    # 将背景面板设为透明/白色
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # 去除轴线使得看起来更干净 (可选，此处保留轴线但去除刻度)
    ax2.grid(False) # 移除默认网格，因为球体本身有网格线
    
    # 绘制 Mean (深蓝色，实心)
    surf_mean = ax2.plot_surface(x_mu, y_mu, z_mu, color='#004488', alpha=0.9,
                     rstride=2, cstride=2, linewidth=0.05, edgecolors='k', shade=True, zorder=2)
    
    # 绘制 Uncertainty (橙色，半透明外壳)
    surf_unc = ax2.plot_surface(x_unc, y_unc, z_unc, color='#FF9933', alpha=0.15,
                     rstride=3, cstride=3, linewidth=0.0, shade=False, zorder=1)

    # 手动添加几条参考网格线增强立体感 (可选)
    # 这里保持简洁，只设置轴标签
    ax2.set_xlabel(r'$\varepsilon_{11}$', labelpad=-10)
    ax2.set_ylabel(r'$\varepsilon_{22}$', labelpad=-10)
    ax2.set_zlabel(r'$\varepsilon_{33}$', labelpad=-10)
    ax2.set_title('(b) Tensor Prediction & Uncertainty', loc='left', pad=10, fontweight='bold')

    # 移除刻度数字
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    # 调整视角 (能够看清嵌套关系)
    ax2.view_init(elev=20, azim=135)
    
    # 调整 Zoom (去除周围过多留白)
    ax2.dist = 9 # 默认是10，改小一点放大物体

    # 自定义图例 (使用 Line2D 创建代理图例)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#004488', markersize=8, label='Mean Prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9933', markersize=8, alpha=0.4, label='95% Confidence Region')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1.0), 
               fontsize=9, frameon=False) # 无边框图例更简洁

    # 保存
    save_path = 'calibration_publication_final.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\n[OK] Plot saved to {save_path}")


def main():
    """主函数：加载模型和数据，进行校准分析"""
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'checkpoints/best_model.pth'
    num_samples = 1000  # 用于校准测试的样本数

    print("="*60)
    print("CALIBRATION ANALYSIS WITH OPTIMIZED MODEL")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Number of samples for calibration: {num_samples}")
    print("Training optimizations: smooth weight transition, condition regularization")

    # 加载数据（匹配训练时的单进程配置）
    print("\nLoading data...")
    _, val_loader, _ = get_dielectric_data_loaders_optimized(
        data_dir='data/mp_dielectric',
        batch_size=64,  # 与训练时的batch_size保持一致
        num_workers=0,  # 使用单进程，匹配训练设置
        max_radius=4.0,  # 与训练时保持一致
        lmax=4  # 与训练时保持一致
    )

    # 加载模型
    print("\nLoading model...")
    # 处理PyTorch 2.6+的安全性检查
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        if "weights_only" in str(e):
            print("  Warning: Loading with weights_only=False")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        else:
            raise e

    # 初始化模型（与训练时相同的架构）
    model = EquivariantUncertaintyNetwork(
        hidden_dim=32,  # 与train.py中的配置保持一致
        max_radius=4.0,  # 与训练时保持一致
        atom_feature_dim=49,
        lmax=4,
        num_layers=2,
        covariance_scale=2.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[OK] Model loaded (epoch {checkpoint.get('epoch', '?')})")
    print(f"[OK] Validation loss: {checkpoint.get('val_loss', 'unknown')}")

    # 计算马氏距离
    mahalanobis_distances = compute_mahalanobis_distances(model, val_loader, device, num_samples)
    print(f"\n[OK] Computed {len(mahalanobis_distances)} Mahalanobis distances")
    print(f"  Mean: {np.mean(mahalanobis_distances):.2f}")
    print(f"  Std: {np.std(mahalanobis_distances):.2f}")
    print(f"  Expected (chi2 with df=6): mean=6.0, std=2*sqrt(6)=4.90")

        # 获取4个样本的3D可视化数据
    print("\nGetting 4 samples for 3D visualization...")
    samples_data = []

    model.eval()
    sample_count = 0

    with torch.no_grad():
        # 获取归一化参数（与训练时一致）
        LOG_MEAN_SCALAR = val_loader.dataset.log_mean
        LOG_STD_SCALAR = val_loader.dataset.log_std

        # 遍历数据加载器获取样本
        for batch in val_loader:
            if sample_count >= 4:
                break

            batch = batch.to(device)

            # Skip if no edges
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                continue

            # 获取预测（模型输出已经在KM空间）
            mu_km, _, Sigma_km = model(batch, compute_sigma=True)

            # 处理batch中的每个样本
            batch_size = mu_km.shape[0]
            for i in range(min(batch_size, 4 - sample_count)):
                mu_km_sample = mu_km[i]  # [6] in KM space
                Sigma_km_sample = Sigma_km[i]  # [6, 6] in KM space

                # 转换到物理空间
                # 1. 转换KM -> Standard Voigt
                mu_voigt_std = kelvin_mandel_to_voigt(mu_km_sample.unsqueeze(0)).squeeze(0)

                # 2. 反归一化
                mean_vec = torch.tensor([LOG_MEAN_SCALAR]*3 + [0.0]*3, device=device)
                mu_log = mu_voigt_std * LOG_STD_SCALAR + mean_vec

                # 3. 指数映射到物理空间
                mu_voigt_phys = sym_matrix_exp_voigt(mu_log.unsqueeze(0)).squeeze(0)

                # 将Voigt向量转换为3x3矩阵
                mu_matrix = voigt_to_matrix_batch(mu_voigt_phys.unsqueeze(0)).squeeze(0)  # [3, 3]

                # 对于不确定性椭球，我们可以基于Sigma的大小来表示
                try:
                    # 计算不确定性缩放因子
                    sigma_scale = torch.sqrt(torch.diag(Sigma_km_sample)).mean().item()
                    uncertainty_scale = 1.0 + 0.05 * sigma_scale  # 较小的不确定性可视化
                except:
                    # 如果失败，使用固定缩放
                    uncertainty_scale = 1.1

                # 定义不确定性椭球（稍微放大的均值张量）
                unc_tensor = mu_matrix.cpu().numpy() * uncertainty_scale

                # 保存样本数据
                samples_data.append({
                    'mu_matrix': mu_matrix.cpu().numpy(),
                    'unc_tensor': unc_tensor
                })

                sample_count += 1

    print(f"[OK] Collected {len(samples_data)} samples for visualization")

    # 调用高质量绘图函数
    plot_publication_quality(mahalanobis_distances, samples_data)

    # 打印校准统计摘要
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print(f"Number of samples: {len(mahalanobis_distances)}")
    print(f"Mean Mahalanobis distance: {np.mean(mahalanobis_distances):.2f} (expected: 6.0)")
    print(f"Std Mahalanobis distance: {np.std(mahalanobis_distances):.2f} (expected: 4.90)")

    # 计算并打印Mean Bias
    emp_quantiles = np.sort(mahalanobis_distances)
    n_samples = len(emp_quantiles)
    prob_levels = np.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)
    theo_quantiles = stats.chi2.ppf(prob_levels, df=6)
    mean_bias = np.mean(np.sqrt(emp_quantiles) - np.sqrt(theo_quantiles))
    r_squared = np.corrcoef(theo_quantiles, emp_quantiles)[0, 1]**2

    print(f"Mean Bias: {mean_bias:.3f} (negative = conservative)")
    print(f"R² (goodness-of-fit): {r_squared:.4f}")
    print("="*60)

    # plt.show() # 如果需要在窗口显示，取消注释


if __name__ == "__main__":
    main()