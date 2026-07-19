# isotropic_utils.py
# ----------------------------------
# Utility functions for isotropic normalization

import torch
import numpy as np


def denormalize_isotropic(tensor_norm, global_mean, global_std):
    """
    Denormalize tensor from isotropic normalization.

    Args:
        tensor_norm: Normalized tensor [6] or [batch_size, 6] in Voigt notation
        global_mean: Global mean of diagonal elements (scalar)
        global_std: Global standard deviation of all elements (scalar)

    Returns:
        Denormalized tensor in same shape
    """
    # Denormalize: T = σ * T_norm + μ * I
    if tensor_norm.dim() == 1:
        # Single tensor - convert from Voigt to 3x3 first
        eps_norm = torch.zeros(3, 3, dtype=tensor_norm.dtype, device=tensor_norm.device)
        eps_norm[0, 0] = tensor_norm[0]
        eps_norm[1, 1] = tensor_norm[1]
        eps_norm[2, 2] = tensor_norm[2]
        eps_norm[1, 2] = tensor_norm[3]
        eps_norm[0, 2] = tensor_norm[4]
        eps_norm[0, 1] = tensor_norm[5]
        eps_norm[2, 1] = tensor_norm[3]  # Symmetric
        eps_norm[2, 0] = tensor_norm[4]
        eps_norm[1, 0] = tensor_norm[5]

        # Denormalize: T = σ * T_norm + μ * I
        I = torch.eye(3, dtype=tensor_norm.dtype, device=tensor_norm.device)
        eps_denorm = eps_norm * global_std + global_mean * I

        # Convert back to Voigt
        tensor_denorm = torch.zeros(6, dtype=tensor_norm.dtype, device=tensor_norm.device)
        tensor_denorm[0] = eps_denorm[0, 0]
        tensor_denorm[1] = eps_denorm[1, 1]
        tensor_denorm[2] = eps_denorm[2, 2]
        tensor_denorm[3] = eps_denorm[1, 2]
        tensor_denorm[4] = eps_denorm[0, 2]
        tensor_denorm[5] = eps_denorm[0, 1]

        return tensor_denorm
    else:
        # Batch of tensors
        batch_size = tensor_norm.shape[0]
        device = tensor_norm.device

        # Convert from Voigt to 3x3 matrices
        eps_norm = torch.zeros(batch_size, 3, 3, dtype=tensor_norm.dtype, device=device)
        eps_norm[:, 0, 0] = tensor_norm[:, 0]
        eps_norm[:, 1, 1] = tensor_norm[:, 1]
        eps_norm[:, 2, 2] = tensor_norm[:, 2]
        eps_norm[:, 1, 2] = tensor_norm[:, 3]
        eps_norm[:, 0, 2] = tensor_norm[:, 4]
        eps_norm[:, 0, 1] = tensor_norm[:, 5]
        eps_norm[:, 2, 1] = tensor_norm[:, 3]  # Symmetric
        eps_norm[:, 2, 0] = tensor_norm[:, 4]
        eps_norm[:, 1, 0] = tensor_norm[:, 5]

        # Create identity matrix
        I = torch.eye(3, dtype=tensor_norm.dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Denormalize: T = σ * T_norm + μ * I
        eps_denorm = eps_norm * global_std + global_mean * I

        # Back to Voigt
        tensor_denorm = torch.zeros(batch_size, 6, dtype=tensor_norm.dtype, device=device)
        tensor_denorm[:, 0] = eps_denorm[:, 0, 0]
        tensor_denorm[:, 1] = eps_denorm[:, 1, 1]
        tensor_denorm[:, 2] = eps_denorm[:, 2, 2]
        tensor_denorm[:, 3] = eps_denorm[:, 1, 2]
        tensor_denorm[:, 4] = eps_denorm[:, 0, 2]
        tensor_denorm[:, 5] = eps_denorm[:, 0, 1]

        return tensor_denorm


def voigt_to_matrix_batch(voigt_vectors):
    """Convert batch of Voigt notation vectors to 3x3 matrices."""
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


def compute_isotropic_stats_batch(tensors_voigt):
    """
    Compute isotropic statistics for a batch of tensors.

    Args:
        tensors_voigt: [batch_size, 6] in Voigt notation

    Returns:
        mean_diag: Mean of all diagonal elements
        std_all: Standard deviation of all elements
    """
    # Extract diagonal elements (first 3)
    diagonal_elements = tensors_voigt[:, :3]

    # Compute mean of diagonal
    mean_diag = diagonal_elements.mean().item()

    # Compute std of all elements
    std_all = tensors_voigt.std().item()

    return mean_diag, std_all


if __name__ == "__main__":
    # Test the functions
    # Example with normalized data
    tensor_norm = torch.tensor([0.5, 1.0, 1.5, 0.1, -0.1, 0.2])
    global_mean = 15.1865
    global_std = 20.0  # Example std

    # Denormalize
    tensor_denorm = denormalize_isotropic(tensor_norm, global_mean, global_std)
    print(f"Denormalized tensor: {tensor_denorm}")

    # Check if it's physically reasonable
    print(f"Diagonal elements: {tensor_denorm[:3]}")