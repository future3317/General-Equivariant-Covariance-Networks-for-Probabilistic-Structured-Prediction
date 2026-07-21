"""
voigt_utils.py
-------------
Utilities for Voigt representation and E(3)-equivariant transformations.
"""
import torch
import numpy as np


def get_voigt_index_map():
    """Mapping from (i,j) to Voigt index."""
    return {
        (0, 0): 0,  # xx
        (1, 1): 1,  # yy
        (2, 2): 2,  # zz
        (1, 2): 3,  # yz
        (0, 2): 4,  # xz
        (0, 1): 5,  # xy
    }


def tensor_to_voigt(C):
    """
    Convert 3x3 symmetric tensor to 6D Voigt vector.

    Args:
        C: [..., 3, 3] symmetric tensor

    Returns:
        voigt: [..., 6] Voigt vector [C11, C22, C33, C23, C13, C12]
    """
    # Ensure symmetry
    C = 0.5 * (C + C.transpose(-2, -1))

    return torch.stack([
        C[..., 0, 0],
        C[..., 1, 1],
        C[..., 2, 2],
        C[..., 1, 2],
        C[..., 0, 2],
        C[..., 0, 1],
    ], dim=-1)


def tensor_to_kelvin_mandel(C):
    """
    Convert 3x3 symmetric tensor to 6D Kelvin-Mandel vector.
    In Kelvin-Mandel notation, shear components are multiplied by sqrt(2)
    to make the rotation matrix orthogonal.

    Args:
        C: [..., 3, 3] symmetric tensor

    Returns:
        km: [..., 6] Kelvin-Mandel vector [C11, C22, C33, sqrt(2)*C23, sqrt(2)*C13, sqrt(2)*C12]
    """
    # Ensure symmetry
    C = 0.5 * (C + C.transpose(-2, -1))

    sqrt2 = torch.sqrt(torch.tensor(2.0, device=C.device, dtype=C.dtype))
    return torch.stack([
        C[..., 0, 0],
        C[..., 1, 1],
        C[..., 2, 2],
        sqrt2 * C[..., 1, 2],
        sqrt2 * C[..., 0, 2],
        sqrt2 * C[..., 0, 1],
    ], dim=-1)


def tensor_to_kelvin_mandel_log(C):
    """
    Direct path: 3x3 symmetric tensor -> Matrix Log -> Kelvin-Mandel vector.
    This is the most efficient and numerically stable path for tensor log operations.

    Args:
        C: [..., 3, 3] symmetric positive definite tensor

    Returns:
        km_log: [..., 6] Kelvin-Mandel vector of matrix logarithm
                 [log(C)11, log(C)22, log(C)33, sqrt(2)*log(C)23, sqrt(2)*log(C)13, sqrt(2)*log(C)12]
                 Note: For off-diagonal elements, we compute log(C) first, then extract the components
    """
    # Ensure symmetry
    C = 0.5 * (C + C.transpose(-2, -1))

    # Use double precision for eigenvalue decomposition (critical for stability)
    orig_dtype = C.dtype
    C_d = C.double()

    # Eigenvalue decomposition
    L, Q = torch.linalg.eigh(C_d)
    # L: [..., 3] eigenvalues, Q: [..., 3, 3] eigenvectors

    # Logarithm of eigenvalues
    L_log = torch.log(torch.clamp(L, min=1e-6))

    # Reconstruct log matrix: log(C) = Q * diag(log(L)) * Q^T
    C_log = Q @ torch.diag_embed(L_log) @ Q.transpose(-2, -1)
    C_log = C_log.to(orig_dtype)

    # Direct conversion to Kelvin-Mandel (no intermediate Voigt)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=C.device, dtype=C.dtype))
    return torch.stack([
        C_log[..., 0, 0],
        C_log[..., 1, 1],
        C_log[..., 2, 2],
        sqrt2 * C_log[..., 1, 2],
        sqrt2 * C_log[..., 0, 2],
        sqrt2 * C_log[..., 0, 1],
    ], dim=-1)


def voigt_to_kelvin_mandel(voigt):
    """
    Convert Voigt vector to Kelvin-Mandel vector.
    Shear components are multiplied by sqrt(2).

    Args:
        voigt: [..., 6] Voigt vector [C11, C22, C33, C23, C13, C12]

    Returns:
        km: [..., 6] Kelvin-Mandel vector [C11, C22, C33, sqrt(2)*C23, sqrt(2)*C13, sqrt(2)*C12]
    """
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=voigt.device, dtype=voigt.dtype))
    km = voigt.clone()
    km[..., 3:] *= sqrt2
    return km


def kelvin_mandel_to_voigt(km):
    """
    Convert Kelvin-Mandel vector to Voigt vector.
    Shear components are divided by sqrt(2).

    Args:
        km: [..., 6] Kelvin-Mandel vector [C11, C22, C33, sqrt(2)*C23, sqrt(2)*C13, sqrt(2)*C12]

    Returns:
        voigt: [..., 6] Voigt vector [C11, C22, C33, C23, C13, C12]
    """
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=km.device, dtype=km.dtype))
    voigt = km.clone()
    voigt[..., 3:] /= sqrt2
    return voigt


def voigt_to_tensor(voigt):
    """
    Convert 6D Voigt vector to 3x3 symmetric tensor.

    Args:
        voigt: [..., 6] Voigt vector [C11, C22, C33, C23, C13, C12]

    Returns:
        C: [..., 3, 3] symmetric tensor
    """
    C = torch.zeros(voigt.shape[:-1] + (3, 3), device=voigt.device, dtype=voigt.dtype)

    C[..., 0, 0] = voigt[..., 0]
    C[..., 1, 1] = voigt[..., 1]
    C[..., 2, 2] = voigt[..., 2]
    C[..., 1, 2] = voigt[..., 3]
    C[..., 2, 1] = voigt[..., 3]
    C[..., 0, 2] = voigt[..., 4]
    C[..., 2, 0] = voigt[..., 4]
    C[..., 0, 1] = voigt[..., 5]
    C[..., 1, 0] = voigt[..., 5]

    return C


def voigt_to_matrix_batch(voigt_vectors):
    """
    Convert batch of Voigt vectors to 3x3 symmetric matrices.
    Compatible with train.py implementation.

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


def sym_matrix_log_voigt(voigt_tensor):
    """
    计算对称矩阵的对数 (Voigt -> Matrix -> Log -> Voigt)
    保持 E(3) 等变性。

    Args:
        voigt_tensor: [B, 6] Voigt representation of symmetric positive definite matrices

    Returns:
        log_voigt: [B, 6] Voigt representation of matrix logarithm
    """
    # 1. Voigt -> 3x3 Matrix
    matrices = voigt_to_matrix_batch(voigt_tensor)  # [B, 3, 3]

    # 2. 特征值分解 (使用 double 精度以防崩溃)
    orig_dtype = matrices.dtype
    matrices_d = matrices.double()

    # 确保完全对称以防数值噪音
    matrices_d = 0.5 * (matrices_d + matrices_d.transpose(-2, -1))

    L, Q = torch.linalg.eigh(matrices_d)
    # L: [B, 3] 特征值, Q: [B, 3, 3] 特征向量

    # 3. 对特征值取对数
    # 加上 epsilon 防止 log(0)
    L_log = torch.log(torch.clamp(L, min=1e-6))

    # 4. 重建矩阵: log(M) = Q * diag(log(L)) * Q^T
    matrices_log = Q @ torch.diag_embed(L_log) @ Q.transpose(-2, -1)

    # 5. Matrix -> Voigt
    # 提取下三角部分
    # xx, yy, zz, yz, xz, xy
    res = torch.stack([
        matrices_log[:, 0, 0],
        matrices_log[:, 1, 1],
        matrices_log[:, 2, 2],
        matrices_log[:, 1, 2],
        matrices_log[:, 0, 2],
        matrices_log[:, 0, 1]
    ], dim=1).to(orig_dtype)

    return res


def sym_matrix_exp_voigt(voigt_tensor):
    """
    计算对称矩阵的指数 (Voigt -> Matrix -> Exp -> Voigt)
    用于将预测值还原为物理量。

    Args:
        voigt_tensor: [B, 6] Voigt representation of matrices

    Returns:
        exp_voigt: [B, 6] Voigt representation of matrix exponential
    """
    matrices = voigt_to_matrix_batch(voigt_tensor)
    orig_dtype = matrices.dtype

    # 使用 double 精度
    matrices_d = matrices.double()
    matrices_d = 0.5 * (matrices_d + matrices_d.transpose(-2, -1))

    L, Q = torch.linalg.eigh(matrices_d)

    # 指数变换
    L_exp = torch.exp(torch.clamp(L, max=20.0))  # 防止溢出

    matrices_exp = Q @ torch.diag_embed(L_exp) @ Q.transpose(-2, -1)

    res = torch.stack([
        matrices_exp[:, 0, 0],
        matrices_exp[:, 1, 1],
        matrices_exp[:, 2, 2],
        matrices_exp[:, 1, 2],
        matrices_exp[:, 0, 2],
        matrices_exp[:, 0, 1]
    ], dim=1).to(orig_dtype)

    return res


def get_voigt_rotation_matrix(R):
    """
    Get the 6x6 rotation matrix for Voigt representation.

    For a rotation matrix R (3x3), the Voigt rotation matrix ρ_c(R) transforms
    the 6D Voigt vector according to: c' = ρ_c(R) * c

    Note: This returns the rotation matrix in standard Voigt notation.
    For orthogonal transformation, use get_kelvin_mandel_rotation_matrix.

    Args:
        R: [..., 3, 3] rotation matrix

    Returns:
        rho: [..., 6, 6] Voigt rotation matrix
    """
    # Ensure R has proper shape
    is_single = R.dim() == 2
    if is_single:
        R = R.unsqueeze(0)
    batch_shape = R.shape[:-2]

    # Use the efficient symbolic formula for all cases
    # This avoids the O(3⁴) nested loops and works efficiently for both batch and single

    # Initialize the 6x6 rotation matrix
    rho = torch.zeros(batch_shape + (6, 6), device=R.device, dtype=R.dtype)

    # Precompute all necessary products
    # Diagonal elements
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # Row 0 (xx component)
    rho[..., 0, 0] = R00 * R00
    rho[..., 0, 1] = R01 * R01
    rho[..., 0, 2] = R02 * R02
    rho[..., 0, 3] = 2 * R01 * R02
    rho[..., 0, 4] = 2 * R00 * R02
    rho[..., 0, 5] = 2 * R00 * R01

    # Row 1 (yy component)
    rho[..., 1, 0] = R10 * R10
    rho[..., 1, 1] = R11 * R11
    rho[..., 1, 2] = R12 * R12
    rho[..., 1, 3] = 2 * R11 * R12
    rho[..., 1, 4] = 2 * R10 * R12
    rho[..., 1, 5] = 2 * R10 * R11

    # Row 2 (zz component)
    rho[..., 2, 0] = R20 * R20
    rho[..., 2, 1] = R21 * R21
    rho[..., 2, 2] = R22 * R22
    rho[..., 2, 3] = 2 * R21 * R22
    rho[..., 2, 4] = 2 * R20 * R22
    rho[..., 2, 5] = 2 * R20 * R21

    # Row 3 (yz component)
    rho[..., 3, 0] = R10 * R20
    rho[..., 3, 1] = R11 * R21
    rho[..., 3, 2] = R12 * R22
    rho[..., 3, 3] = R11 * R22 + R12 * R21
    rho[..., 3, 4] = R10 * R22 + R12 * R20
    rho[..., 3, 5] = R10 * R21 + R11 * R20

    # Row 4 (xz component)
    rho[..., 4, 0] = R00 * R20
    rho[..., 4, 1] = R01 * R21
    rho[..., 4, 2] = R02 * R22
    rho[..., 4, 3] = R01 * R22 + R02 * R21
    rho[..., 4, 4] = R00 * R22 + R02 * R20
    rho[..., 4, 5] = R00 * R21 + R01 * R20

    # Row 5 (xy component)
    rho[..., 5, 0] = R00 * R10
    rho[..., 5, 1] = R01 * R11
    rho[..., 5, 2] = R02 * R12
    rho[..., 5, 3] = R01 * R12 + R02 * R11
    rho[..., 5, 4] = R00 * R12 + R02 * R10
    rho[..., 5, 5] = R00 * R11 + R01 * R10

    # Squeeze if single rotation
    if is_single:
        return rho.squeeze(0)

    # Optional: validate invertibility in batch mode for debugging
    if __debug__ and batch_shape == (1,):  # Only validate for small batches
        dets = torch.linalg.det(rho)
        assert torch.all(torch.abs(dets) > 1e-6), f"rho should be invertible, got det={dets}"

    return rho


def get_kelvin_mandel_rotation_matrix(R):
    """
    Get the 6x6 ORTHOGONAL rotation matrix for Kelvin-Mandel representation.

    In Kelvin-Mandel notation, the rotation matrix is orthogonal (ρ @ ρ.T = I).
    This is achieved by scaling shear components by sqrt(2).

    Args:
        R: [..., 3, 3] rotation matrix

    Returns:
        rho_km: [..., 6, 6] ORTHOGONAL Kelvin-Mandel rotation matrix
    """
    # Get the standard Voigt rotation matrix
    rho = get_voigt_rotation_matrix(R)

    # Convert to Kelvin-Mandel scaling
    # For Kelvin-Mandel representation, if v = B*k where B = diag(1,1,1,√2,√2,√2)
    # then the rotation matrix transforms as: ρ_KM = B @ ρ_Voigt @ B^-1
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=R.device, dtype=R.dtype))

    # Create transformation matrix B = diag(1,1,1,√2,√2,√2)
    B = torch.eye(6, device=R.device, dtype=R.dtype)
    B[3, 3] = sqrt2
    B[4, 4] = sqrt2
    B[5, 5] = sqrt2

    # Create inverse transformation matrix B^-1 = diag(1,1,1,1/√2,1/√2,1/√2)
    B_inv = torch.eye(6, device=R.device, dtype=R.dtype)
    B_inv[3, 3] = 1.0 / sqrt2
    B_inv[4, 4] = 1.0 / sqrt2
    B_inv[5, 5] = 1.0 / sqrt2

    # Apply the similarity transformation: ρ_KM = B @ ρ_Voigt @ B^-1
    rho_km = B @ rho @ B_inv

    # [调试] 检查 KM 旋转矩阵的正交性
    # KM 空间的旋转矩阵应该是正交的：ρ_KM @ ρ_KM^T = I
    if __debug__:
        # 验证正交性：ρ @ ρ.T ≈ I
        identity_check = rho_km @ rho_km.transpose(-2, -1)
        identity = torch.eye(6, device=R.device, dtype=R.dtype)

        # 对于 batch 维度适配
        if rho_km.dim() == 3:
            identity = identity.unsqueeze(0).expand(rho_km.shape[0], -1, -1)

        max_error = (identity_check - identity).abs().max().item()
        assert max_error < 1e-5, f"KM rotation matrix not orthogonal: max error = {max_error:.2e}"

    return rho_km


def random_rotation_matrix(batch_size=1, device='cpu', squeeze_single=False):
    """
    Generate random rotation matrices with numerical stability guarantees.

    Args:
        batch_size: Number of rotation matrices to generate
        device: Device to place the tensors on
        squeeze_single: If True, squeeze the batch dimension when batch_size=1
                       (default: False for consistent batch dimension)

    Returns:
        R: [batch_size, 3, 3] orthogonal rotation matrices (det(R) = 1)
           If squeeze_single=True and batch_size=1, returns [3, 3]
    """
    # Generate random quaternions using uniform distribution on S^3
    u1, u2, u3 = torch.rand(3, batch_size, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

    # Convert to rotation matrix
    R = torch.zeros(batch_size, 3, 3, device=device)

    R[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    R[:, 0, 1] = 2 * (q2 * q3 - q1 * q4)
    R[:, 0, 2] = 2 * (q2 * q4 + q1 * q3)

    R[:, 1, 0] = 2 * (q2 * q3 + q1 * q4)
    R[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    R[:, 1, 2] = 2 * (q3 * q4 - q1 * q2)

    R[:, 2, 0] = 2 * (q2 * q4 - q1 * q3)
    R[:, 2, 1] = 2 * (q3 * q4 + q1 * q2)
    R[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)

    # Numerical stability: orthogonalize using QR decomposition
    # This ensures R @ R.T = I and det(R) = 1 even with floating point errors
    if batch_size > 0:
        # Process each rotation matrix individually for stability
        for i in range(batch_size):
            # QR decomposition guarantees orthogonality
            Q, _ = torch.linalg.qr(R[i])

            # Ensure proper rotation (det = +1, not -1)
            det = torch.det(Q)
            if det < 0:
                # If det = -1, flip one column to make it +1
                Q[:, 0] = -Q[:, 0]

            R[i] = Q

    # [修复] 统一返回维度，根据参数决定是否 squeeze
    if squeeze_single and batch_size == 1:
        return R.squeeze(0)

    return R


def test_voigt_transformations():
    """Test the Voigt transformation functions."""
    print("Testing Voigt transformation utilities...")

    # Test tensor to Voigt and back
    C = torch.randn(3, 3)
    C = 0.5 * (C + C.T)  # Make symmetric

    voigt = tensor_to_voigt(C)
    C_reconstructed = voigt_to_tensor(voigt)

    assert torch.allclose(C, C_reconstructed, atol=1e-6), "Tensor-Voigt conversion failed"
    print("PASS: Tensor-Voigt conversion works correctly")

    # Test single rotation matrix
    print("\nTesting single rotation matrix...")
    R = random_rotation_matrix(batch_size=1, squeeze_single=True)
    rho = get_voigt_rotation_matrix(R)

    # Verify orthogonality
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6), "R should be orthogonal"
    assert torch.abs(torch.det(R) - 1.0) < 1e-6, "det(R) should be 1"
    print("PASS: Single rotation matrix is orthogonal with det=1")

    # Rotate the tensor directly
    C_rot = R @ C @ R.mT
    voigt_rot = tensor_to_voigt(C_rot)

    # Rotate the Voigt vector
    voigt_rot2 = rho @ voigt

    assert torch.allclose(voigt_rot, voigt_rot2, atol=1e-6), "Voigt rotation failed"
    print("PASS: Single Voigt rotation works correctly")

    # Check that rho is invertible
    det = torch.linalg.det(rho)
    assert torch.abs(det) > 1e-6, "rho should be invertible"
    print(f"PASS: Single Voigt rotation matrix is invertible (det = {det.item():.4f})")

    # Test KM rotation matrix orthogonality
    rho_km = get_kelvin_mandel_rotation_matrix(R)
    identity_km = rho_km @ rho_km.T
    assert torch.allclose(identity_km, torch.eye(6), atol=1e-6), "KM rotation matrix should be orthogonal"
    print("PASS: Single KM rotation matrix is orthogonal")

    # Test batch rotation matrices
    print("\nTesting batch rotation matrices...")
    batch_size = 4
    R_batch = random_rotation_matrix(batch_size=batch_size)
    rho_batch = get_voigt_rotation_matrix(R_batch)

    # Verify orthogonality for all matrices in batch
    for i in range(batch_size):
        assert torch.allclose(R_batch[i] @ R_batch[i].T, torch.eye(3), atol=1e-6), f"R[{i}] should be orthogonal"
        assert torch.abs(torch.det(R_batch[i]) - 1.0) < 1e-6, f"det(R[{i}]) should be 1"

        # Check rho invertibility
        det_rho = torch.linalg.det(rho_batch[i])
        assert torch.abs(det_rho) > 1e-6, f"rho[{i}] should be invertible"

    print(f"PASS: All {batch_size} rotation matrices are orthogonal with det=1")
    print(f"PASS: All {batch_size} Voigt rotation matrices are invertible")

    # Batch rotation invariance test
    C_batch = C.unsqueeze(0).repeat(batch_size, 1, 1)
    C_rot_batch = torch.bmm(torch.bmm(R_batch, C_batch), R_batch.transpose(-2, -1))
    voigt_batch = tensor_to_voigt(C_batch)
    voigt_rot_batch = tensor_to_voigt(C_rot_batch)
    voigt_rot_batch2 = torch.bmm(rho_batch, voigt_batch.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(voigt_rot_batch, voigt_rot_batch2, atol=1e-6), "Batch Voigt rotation failed"
    print("PASS: Batch Voigt rotation works correctly")

    print("\nAll tests passed!")


def test_tensor_to_km_log():
    """Test the new tensor_to_kelvin_mandel_log function."""
    print("\nTesting tensor_to_kelvin_mandel_log...")

    # Create a test SPD matrix
    C = torch.randn(3, 3)
    C = C @ C.T + torch.eye(3) * 0.1  # Make it SPD

    # Method 1: Direct tensor -> log -> KM
    km_log_direct = tensor_to_kelvin_mandel_log(C.unsqueeze(0)).squeeze(0)

    # Method 2: Traditional path: tensor -> voigt -> log -> KM
    voigt = tensor_to_voigt(C)
    voigt_log = sym_matrix_log_voigt(voigt.unsqueeze(0)).squeeze(0)
    km_log_traditional = voigt_to_kelvin_mandel(voigt_log)

    # Compare results
    assert torch.allclose(km_log_direct, km_log_traditional, atol=1e-5), \
        f"Direct and traditional methods differ: {km_log_direct - km_log_traditional}"

    print("PASS: tensor_to_kelvin_mandel_log produces same results as traditional path")
    print(f"  Direct method:    {km_log_direct}")
    print(f"  Traditional:      {km_log_traditional}")
    print(f"  Max difference:   {(km_log_direct - km_log_traditional).abs().max().item():.2e}")


if __name__ == "__main__":
    test_voigt_transformations()
    test_tensor_to_km_log()
