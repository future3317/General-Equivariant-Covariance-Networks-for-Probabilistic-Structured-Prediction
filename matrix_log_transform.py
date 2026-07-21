"""
matrix_log_transform.py
-----------------------
Matrix logarithm and exponential transformations for dielectric tensors.
These transformations preserve E(3) equivariance while improving training stability.
"""
import torch
from voigt_utils import voigt_to_tensor


def voigt_to_matrix_batch(voigt_vectors):
    """Convert batch of Voigt notation vectors to 3x3 matrices.

    Args:
        voigt_vectors: Tensor of shape [batch_size, 6] in Voigt notation
                      [c11, c22, c33, c23, c13, c12]

    Returns:
        Tensor of shape [batch_size, 3, 3] representing symmetric 3x3 matrices
    """
    batch_size = voigt_vectors.shape[0]
    matrices = torch.zeros(batch_size, 3, 3, device=voigt_vectors.device, dtype=voigt_vectors.dtype)

    # Fill diagonal components
    matrices[:, 0, 0] = voigt_vectors[:, 0]  # c11
    matrices[:, 1, 1] = voigt_vectors[:, 1]  # c22
    matrices[:, 2, 2] = voigt_vectors[:, 2]  # c33

    # Fill off-diagonal components (symmetric)
    matrices[:, 1, 2] = voigt_vectors[:, 3]  # c23
    matrices[:, 2, 1] = voigt_vectors[:, 3]  # c23
    matrices[:, 0, 2] = voigt_vectors[:, 4]  # c13
    matrices[:, 2, 0] = voigt_vectors[:, 4]  # c13
    matrices[:, 0, 1] = voigt_vectors[:, 5]  # c12
    matrices[:, 1, 0] = voigt_vectors[:, 5]  # c12

    return matrices


def matrix_to_voigt_batch(matrices):
    """Convert batch of 3x3 symmetric matrices to Voigt notation.

    Args:
        matrices: Tensor of shape [batch_size, 3, 3] symmetric matrices

    Returns:
        Tensor of shape [batch_size, 6] in Voigt notation [c11, c22, c33, c23, c13, c12]
    """
    # Ensure symmetry
    matrices = 0.5 * (matrices + matrices.transpose(-2, -1))

    batch_size = matrices.shape[0]
    voigt = torch.zeros(batch_size, 6, device=matrices.device, dtype=matrices.dtype)

    # Extract components
    voigt[:, 0] = matrices[:, 0, 0]  # c11
    voigt[:, 1] = matrices[:, 1, 1]  # c22
    voigt[:, 2] = matrices[:, 2, 2]  # c33
    voigt[:, 3] = matrices[:, 1, 2]  # c23
    voigt[:, 4] = matrices[:, 0, 2]  # c13
    voigt[:, 5] = matrices[:, 0, 1]  # c12

    return voigt


def matrix_logarithm_transform(targets_voigt, epsilon=1e-4): # [修改1] 默认 epsilon 增大到 1e-4
    """
    Transform physical dielectric tensors to matrix logarithm space.
    This preserves E(3) equivariance and improves training stability.

    Args:
        targets_voigt: [B, 6] Physical dielectric tensors in Voigt notation
        epsilon: Small constant for numerical stability (default: 1e-4)

    Returns:
        targets_log_voigt: [B, 6] Matrix logarithm of dielectric tensors in Voigt
    """
    # 保存原始精度
    orig_dtype = targets_voigt.dtype
    orig_device = targets_voigt.device

    # 1. Convert Voigt to 3x3 matrices and升精度到float64
    matrices = voigt_to_matrix_batch(targets_voigt).double()  # [B, 3, 3]

    # 2. Ensure matrices are symmetric (numerical stability)
    matrices = 0.5 * (matrices + matrices.transpose(-2, -1))

    # 3. Eigenvalue decomposition for symmetric matrices
    # For dielectric tensors, eigenvalues should be positive (> 0)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)

    # 4. [关键修改] 更加激进的 Clamp
    # log(x) 的导数是 1/x。如果特征值太小，梯度会爆炸。
    # 物理上介电常数通常 >= 1.0。这里我们设为 1e-4 作为绝对底线，防止数值噪声导致的崩溃。
    # 1e-4 的对数值约为 -9.2，梯度为 10000，这是 float32 能承受的安全范围。
    eigenvalues = torch.clamp(eigenvalues, min=1e-4)

    # 5. Take logarithm of eigenvalues
    # For dielectric constants, we can use log(eigenvalues) directly
    # since they are physically > 1
    eigenvalues_log = torch.log(eigenvalues)

    # 6. Reconstruct log matrix: V @ diag(log(eigenvalues)) @ V^T
    matrices_log = torch.bmm(
        torch.bmm(eigenvectors, torch.diag_embed(eigenvalues_log)),
        eigenvectors.transpose(-2, -1)
    )

    # 7. Convert back to Voigt notation
    targets_log_voigt = matrix_to_voigt_batch(matrices_log)

    # 8. 转回原始精度（通常为float32）
    return targets_log_voigt.to(dtype=orig_dtype, device=orig_device)


def matrix_exponential_transform(predictions_log_voigt):
    """
    Transform predictions from matrix logarithm space back to physical space.
    This is the inverse of matrix_logarithm_transform.

    Args:
        predictions_log_voigt: [B, 6] Predictions in matrix logarithm space (Voigt)

    Returns:
        predictions_voigt: [B, 6] Physical dielectric tensors in Voigt notation
    """
    # 保存原始精度
    orig_dtype = predictions_log_voigt.dtype
    orig_device = predictions_log_voigt.device

    # 1. Convert Voigt to 3x3 matrices and升精度到float64
    matrices_log = voigt_to_matrix_batch(predictions_log_voigt).double()  # [B, 3, 3]

    # 2. Ensure symmetry
    matrices_log = 0.5 * (matrices_log + matrices_log.transpose(-2, -1))

    # 3. Matrix exponential using eigenvalue decomposition
    # For symmetric matrices: exp(A) = V @ diag(exp(eigenvalues)) @ V^T
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices_log)

    # 4. Exponential of eigenvalues
    eigenvalues_exp = torch.exp(eigenvalues)

    # 5. Reconstruct matrix: V @ diag(exp(eigenvalues)) @ V^T
    matrices = torch.bmm(
        torch.bmm(eigenvectors, torch.diag_embed(eigenvalues_exp)),
        eigenvectors.transpose(-2, -1)
    )

    # 6. Convert back to Voigt notation
    predictions_voigt = matrix_to_voigt_batch(matrices)

    # 7. 转回原始精度（通常为float32）
    return predictions_voigt.to(dtype=orig_dtype, device=orig_device)


def analyze_log_transform_statistics(dataloader):
    """
    Analyze the statistics of dielectric tensors before and after log transform.
    This helps in setting up proper normalization.

    Args:
        dataloader: DataLoader containing the dataset

    Returns:
        dict: Statistics of original and log-transformed data
    """
    all_original = []
    all_log = []

    print("\nAnalyzing dielectric tensor statistics...")
    print("="*60)

    for i, batch in enumerate(dataloader):
        if i >= 20:  # Sample first 20 batches for efficiency
            break

        targets_norm = batch['target']  # Normalized values

        # Denormalize to get physical values
        from isotropic_utils import denormalize_isotropic
        targets_phys = denormalize_isotropic(targets_norm,
                                            dataloader.dataset.global_mean,
                                            dataloader.dataset.global_std)
        all_original.append(targets_phys)

        # Apply log transform
        targets_log_phys = matrix_logarithm_transform(targets_phys)
        all_log.append(targets_log_phys)

    # Concatenate all data
    all_original = torch.cat(all_original, dim=0)
    all_log = torch.cat(all_log, dim=0)

    # Calculate statistics
    stats = {
        'original': {
            'mean': all_original.mean(dim=0),
            'std': all_original.std(dim=0),
            'min': all_original.min(dim=0)[0],
            'max': all_original.max(dim=0)[0],
            'range': all_original.max(dim=0)[0] - all_original.min(dim=0)[0]
        },
        'log': {
            'mean': all_log.mean(dim=0),
            'std': all_log.std(dim=0),
            'min': all_log.min(dim=0)[0],
            'max': all_log.max(dim=0)[0],
            'range': all_log.max(dim=0)[0] - all_log.min(dim=0)[0]
        }
    }

    # Print statistics
    print("\n[Original Physical Values]")
    print("  Diagonal (εxx, εyy, εzz):")
    diag_mean = stats['original']['mean'][:3].numpy()
    diag_std = stats['original']['std'][:3].numpy()
    diag_min = stats['original']['min'][:3].numpy()
    diag_max = stats['original']['max'][:3].numpy()
    print(f"    Mean: [{diag_mean[0]:.3f}, {diag_mean[1]:.3f}, {diag_mean[2]:.3f}]")
    print(f"    Std:  [{diag_std[0]:.3f}, {diag_std[1]:.3f}, {diag_std[2]:.3f}]")
    print(f"    Range: [{diag_min[0]:.1f}, {diag_min[1]:.1f}, {diag_min[2]:.1f}] to [{diag_max[0]:.1f}, {diag_max[1]:.1f}, {diag_max[2]:.1f}]")
    print("  Off-diagonal (εyz, εxz, εxy):")
    off_mean = stats['original']['mean'][3:].numpy()
    off_std = stats['original']['std'][3:].numpy()
    off_min = stats['original']['min'][3:].numpy()
    off_max = stats['original']['max'][3:].numpy()
    print(f"    Mean: [{off_mean[0]:.3f}, {off_mean[1]:.3f}, {off_mean[2]:.3f}]")
    print(f"    Std:  [{off_std[0]:.3f}, {off_std[1]:.3f}, {off_std[2]:.3f}]")
    print(f"    Range: [{off_min[0]:.1f}, {off_min[1]:.1f}, {off_min[2]:.1f}] to [{off_max[0]:.1f}, {off_max[1]:.1f}, {off_max[2]:.1f}]")

    print("\n[After Matrix Log Transform]")
    print("  Diagonal (log εxx, log εyy, log εzz):")
    log_diag_mean = stats['log']['mean'][:3].numpy()
    log_diag_std = stats['log']['std'][:3].numpy()
    log_diag_min = stats['log']['min'][:3].numpy()
    log_diag_max = stats['log']['max'][:3].numpy()
    print(f"    Mean: [{log_diag_mean[0]:.3f}, {log_diag_mean[1]:.3f}, {log_diag_mean[2]:.3f}]")
    print(f"    Std:  [{log_diag_std[0]:.3f}, {log_diag_std[1]:.3f}, {log_diag_std[2]:.3f}]")
    print(f"    Range: [{log_diag_min[0]:.1f}, {log_diag_min[1]:.1f}, {log_diag_min[2]:.1f}] to [{log_diag_max[0]:.1f}, {log_diag_max[1]:.1f}, {log_diag_max[2]:.1f}]")
    print("  Off-diagonal (log εyz, log εxz, log εxy):")
    log_off_mean = stats['log']['mean'][3:].numpy()
    log_off_std = stats['log']['std'][3:].numpy()
    log_off_min = stats['log']['min'][3:].numpy()
    log_off_max = stats['log']['max'][3:].numpy()
    print(f"    Mean: [{log_off_mean[0]:.3f}, {log_off_mean[1]:.3f}, {log_off_mean[2]:.3f}]")
    print(f"    Std:  [{log_off_std[0]:.3f}, {log_off_std[1]:.3f}, {log_off_std[2]:.3f}]")
    print(f"    Range: [{log_off_min[0]:.1f}, {log_off_min[1]:.1f}, {log_off_min[2]:.1f}] to [{log_off_max[0]:.1f}, {log_off_max[1]:.1f}, {log_off_max[2]:.1f}]")

    # Calculate compression ratio
    original_range = stats['original']['range'].mean().item()
    log_range = stats['log']['range'].mean().item()
    compression = original_range / log_range if log_range > 0 else float('inf')

    print("\n[Compression Ratio]")
    print(f"  Average range compression: {compression:.2f}x")
    print(f"  Original avg range: {original_range:.2f}")
    print(f"  Log avg range: {log_range:.2f}")

    return stats


def test_matrix_log_transform():
    """Test the matrix logarithm transform for correctness and equivariance."""
    print("\nTesting matrix logarithm transform...")
    print("="*60)

    # Test 1: Basic round-trip test
    print("\n[Test 1] Round-trip test (log -> exp)")
    torch.manual_seed(42)

    # Create a positive definite tensor (typical dielectric values)
    # Dielectric constants are usually > 1
    base_values = torch.tensor([[4.0, 0.1, 0.2],
                               [0.1, 5.0, 0.1],
                               [0.2, 0.1, 6.0]])

    # Add some random variation
    batch = []
    for i in range(5):
        noise = torch.randn(3, 3) * 0.1
        matrix = base_values + noise
        matrix = 0.5 * (matrix + matrix.T)  # Ensure symmetry
        # Ensure positive definite by making eigenvalues > 1
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=1.0)
        matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        batch.append(matrix)

    batch = torch.stack(batch)
    batch_voigt = matrix_to_voigt_batch(batch)

    # Apply log transform
    log_voigt = matrix_logarithm_transform(batch_voigt)

    # Apply exponential transform
    reconstructed_voigt = matrix_exponential_transform(log_voigt)

    # Check round-trip accuracy
    max_error = torch.max(torch.abs(batch_voigt - reconstructed_voigt)).item()
    mean_error = torch.mean(torch.abs(batch_voigt - reconstructed_voigt)).item()

    print(f"  Max reconstruction error: {max_error:.2e}")
    print(f"  Mean reconstruction error: {mean_error:.2e}")

    if max_error < 1e-5:
        print("  [PASS] Round-trip test PASSED")
    else:
        print("  [FAIL] Round-trip test FAILED")

    # Test 2: Equivariance test
    print("\n[Test 2] E(3) Equivariance test")

    # Create a test tensor
    test_matrix = torch.tensor([[3.0, 0.5, 0.2],
                                [0.5, 4.0, 0.1],
                                [0.2, 0.1, 5.0]], dtype=torch.float32)
    test_voigt = matrix_to_voigt_batch(test_matrix.unsqueeze(0))

    # Apply log transform
    test_log_voigt = matrix_logarithm_transform(test_voigt)

    # Generate a random rotation
    from voigt_utils import random_rotation_matrix
    R = random_rotation_matrix()

    # Method 1: Rotate original tensor, then apply log
    rotated_matrix = R @ test_matrix @ R.T
    rotated_voigt = matrix_to_voigt_batch(rotated_matrix.unsqueeze(0))
    rotated_log_voigt = matrix_logarithm_transform(rotated_voigt)

    # Method 2: Apply log first, then rotate using Voigt rotation
    from voigt_utils import get_voigt_rotation_matrix
    rho = get_voigt_rotation_matrix(R)
    log_rotated_voigt = rho @ test_log_voigt.T

    # Compare the two methods (need to transpose to match shapes)
    equivariance_error = torch.max(torch.abs(rotated_log_voigt - log_rotated_voigt.T)).item()

    print(f"  Equivariance error: {equivariance_error:.2e}")

    if equivariance_error < 1e-5:
        print("  [PASS] Equivariance test PASSED")
    else:
        print("  [FAIL] Equivariance test FAILED")

    # Test 3: Positive definiteness preservation
    print("\n[Test 3] Positive definiteness test")

    # Generate random positive definite matrices
    n_test = 10
    all_positive = True

    for i in range(n_test):
        # Random symmetric matrix
        A = torch.randn(3, 3)
        A = 0.5 * (A + A.T)

        # Make it positive definite
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = torch.abs(eigvals) + 1.0  # Ensure eigenvalues > 1
        A_pos = eigvecs @ torch.diag(eigvals) @ eigvecs.T

        # Convert to Voigt and apply transforms
        A_voigt = matrix_to_voigt_batch(A_pos.unsqueeze(0))
        A_log = matrix_logarithm_transform(A_voigt)
        A_exp = matrix_exponential_transform(A_log)
        A_exp_matrix = voigt_to_tensor(A_exp.squeeze(0))

        # Check if result is still positive definite
        final_eigvals = torch.linalg.eigvalsh(A_exp_matrix)
        if not torch.all(final_eigvals > 0):
            all_positive = False
            print(f"  Failed on test {i}: eigenvalues = {final_eigvals}")

    if all_positive:
        print("  [PASS] Positive definiteness preserved for all test cases")
    else:
        print("  [FAIL] Positive definiteness NOT preserved")

    print("\nAll tests completed!")
    return True


if __name__ == "__main__":
    # Run tests
    test_matrix_log_transform()
