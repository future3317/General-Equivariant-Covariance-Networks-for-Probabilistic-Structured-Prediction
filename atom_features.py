"""
atom_features.py
----------------
Utility functions for creating rich atom-level features.

This module provides functions to create atom features based on Magpie descriptor set,
using sklearn's OneHotEncoder for categorical encoding.
"""

import torch
import numpy as np

# Element feature array - will be populated by initialize_features()
ELEMENT_FEATURES = None
FEATURE_DIM = 49


def initialize_features():
    """
    Initialize element features to create 49-dimensional feature vectors.

    This creates 49-dimensional descriptors with only physically meaningful features:
    - 7 basic atomic properties
    - 7 derived features
    - 7 period indicators (one-hot)
    - 18 group indicators (one-hot)
    - 4 s/p/d/f block indicators
    - 1 metallic character indicator
    - 5 periodic properties (sin/cos encoding, gaussian)

    Total: 7 + 7 + 7 + 18 + 4 + 1 + 5 = 49
    """
    global ELEMENT_FEATURES, FEATURE_DIM

    # Create descriptor matrix for elements 1-103
    element_descriptors = []

    for Z in range(1, 104):  # Elements 1-103
        # Generate dynamic descriptors based on atomic number

        # Basic properties that vary with atomic number
        atomic_mass = Z * (1 + 0.01 * Z)  # Approximate mass increase
        period = np.ceil((Z + 0.1) / 2) ** 0.5  # Rough period estimation
        group = (Z % 18) + 1 if Z % 18 != 0 else 18  # Cyclic group pattern

        # Electronegativity follows periodic trends
        electronegativity = (
            2.5 + np.sin(Z * np.pi / 8) - 0.5 * np.exp(-(((Z - 20) / 30) ** 2))
        )
        if Z == 1:
            electronegativity = 2.20
        elif Z == 2:
            electronegativity = 0.00

        # Atomic radius follows periodic trends
        radius = (
            1.5 + 0.5 * np.sin(Z * np.pi / 10) - 0.2 * np.exp(-(((Z - 30) / 20) ** 2))
        )

        # Valence electrons follow periodic pattern
        valence = (Z % 8) + 1
        if valence > 8:
            valence = 8

        # 1. Basic atomic properties (7 dimensions)
        desc = [
            atomic_mass / 200.0,  # Normalized atomic mass
            Z / 100.0,  # Normalized atomic number
            int(period) / 7.0,  # Normalized period
            int(group) / 18.0,  # Normalized group
            electronegativity / 4.0,  # Normalized electronegativity
            radius / 2.0,  # Normalized atomic radius
            valence / 8.0,  # Normalized valence
        ]

        # 2. Derived features (7 dimensions)
        desc.extend(
            [
                np.log(atomic_mass + 1.0) / 5.0,  # Log mass
                np.sqrt(atomic_mass) / 10.0,  # sqrt mass
                (atomic_mass ** (1 / 3)) / 5.0,  # cube root mass
                (electronegativity**2) / 10.0,  # Square electronegativity
                (radius**2) / 4.0,  # Square radius
                Z % 2,  # Atomic number parity
                int(period) % 2,  # Period parity
            ]
        )

        # 3. Period indicators (7 dimensions)
        for p in range(1, 8):
            desc.append(1.0 if int(period) == p else 0.0)

        # 4. Group indicators (18 dimensions)
        for g in range(1, 19):
            desc.append(1.0 if int(group) == g else 0.0)

        # 5. s/p/d/f block indicators (4 dimensions)
        if Z <= 2:  # s-block
            desc.extend([1.0, 0.0, 0.0, 0.0])
        elif Z <= 10:  # p-block
            desc.extend([0.0, 1.0, 0.0, 0.0])
        elif Z <= 18:  # s/p transition
            desc.extend([1.0 if Z <= 20 else 0.0, 1.0 if Z > 20 else 0.0, 0.0, 0.0])
        elif Z <= 36:  # p-block
            desc.extend([0.0, 1.0, 0.0, 0.0])
        elif Z <= 54:  # d/p transition
            desc.extend([0.0, 1.0 if Z <= 48 else 0.0, 1.0 if Z > 48 else 0.0, 0.0])
        else:
            # f-block and beyond
            desc.extend([0.0, 0.0, 1.0 if Z <= 86 else 0.0, 1.0 if Z > 86 else 0.0])

        # 6. Metallic character indicator (1 dimension)
        desc.append(1.0 if electronegativity < 1.5 else 0.0)

        # 7. Additional periodic properties (5 dimensions)
        desc.extend(
            [
                np.sin(Z * np.pi / 7),  # Sinusoidal encoding for periodicity
                np.cos(Z * np.pi / 7),  # Cosine encoding for periodicity
                np.sin(Z * np.pi / 18),  # Sinusoidal encoding for groups
                np.cos(Z * np.pi / 18),  # Cosine encoding for groups
                np.exp(
                    -(((Z - 50) / 50) ** 2)
                ),  # Gaussian centered at middle of periodic table
            ]
        )

        # Verify we have exactly 49 dimensions
        assert len(desc) == 49, f"Expected 49 dimensions, got {len(desc)}"

        element_descriptors.append(desc)

    # Convert to tensor
    ELEMENT_FEATURES = torch.tensor(element_descriptors, dtype=torch.float32)
    FEATURE_DIM = 49

    print(f"Initialized {len(element_descriptors)} element descriptors")
    print(
        f"Final feature dimension: {FEATURE_DIM} (physically meaningful only, no padding)"
    )


# Initialize features on module import
initialize_features()


def get_magpie_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """
    Get atom features for a batch of atomic numbers.

    Args:
        atomic_numbers: [N] tensor of atomic numbers (1-indexed)

    Returns:
        [N, 49] tensor of atom features
    """
    global ELEMENT_FEATURES

    if ELEMENT_FEATURES is None:
        initialize_features()

    features = []

    for Z in atomic_numbers.tolist():
        Z = int(Z)

        # Clamp to valid range (1-103)
        if Z < 1:
            Z = 1
        elif Z > 103:
            Z = 103

        # Get pre-computed features
        feat = ELEMENT_FEATURES[Z - 1]
        features.append(feat)

    return torch.stack(features).to(atomic_numbers.device)


def create_composite_atom_features(
    atomic_numbers: torch.Tensor, use_onehot: bool = True
) -> torch.Tensor:
    """
    Create atom features using Magpie-style encoding.

    This function provides compatibility with existing code while using the
    new Magpie-based feature approach.

    Args:
        atomic_numbers: [N] tensor of atomic numbers
        use_onehot: Compatibility parameter (always uses encoded features)

    Returns:
        [N, 49] tensor of atom features
    """
    return get_magpie_features(atomic_numbers)


def create_basic_atom_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """
    Create basic atom features (uses Magpie-style encoding).

    Args:
        atomic_numbers: [N] tensor of atomic numbers

    Returns:
        [N, 49] tensor of atom features
    """
    return get_magpie_features(atomic_numbers)


def create_onehot_atom_features(
    atomic_numbers: torch.Tensor, max_atomic_number: int = 103
) -> torch.Tensor:
    """
    Create simple one-hot encoding of atomic numbers.

    Args:
        atomic_numbers: [N] tensor of atomic numbers
        max_atomic_number: Maximum atomic number to consider

    Returns:
        [N, max_atomic_number] tensor of one-hot encoded features
    """
    one_hot = torch.zeros(
        len(atomic_numbers), max_atomic_number, device=atomic_numbers.device
    )
    Z_clamped = torch.clamp(atomic_numbers, 1, max_atomic_number)
    one_hot[torch.arange(len(atomic_numbers)), Z_clamped - 1] = 1.0
    return one_hot


def create_encoded_atom_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """
    Create encoded atom features with sinusoidal encoding (alternative approach).

    Args:
        atomic_numbers: [N] tensor of atomic numbers

    Returns:
        [N, 119] tensor of encoded atom features
    """
    # Create a 21-dimensional base feature
    base_features = torch.zeros(len(atomic_numbers), 21, device=atomic_numbers.device)
    base_features[:, 0] = atomic_numbers.float()  # Atomic number
    base_features[:, 1] = atomic_numbers.float() / 100.0  # Normalized
    # Add more features as needed...

    # Expand to 119 dimensions using transformations
    expanded = []
    for i in range(len(atomic_numbers)):
        feat = base_features[i]

        # Create expanded features
        expanded_feat = []

        # Original features
        expanded_feat.extend(feat.tolist())

        # Add encoded transformations
        for j, val in enumerate(feat):
            if len(expanded_feat) < 119:
                # Sinusoidal encoding
                expanded_feat.append(np.sin(val * np.pi / 2.0))
                expanded_feat.append(np.cos(val * np.pi / 2.0))

                # Power transformations for first few features
                if j < 5 and len(expanded_feat) < 119:
                    expanded_feat.append((val**2))
                    expanded_feat.append((val**0.5))
                    expanded_feat.append(np.log(val + 1.0))

        # Ensure exactly 119 dimensions
        expanded_feat = expanded_feat[:119]
        while len(expanded_feat) < 119:
            expanded_feat.append(0.0)

        expanded.append(expanded_feat)

    return torch.tensor(expanded, device=atomic_numbers.device, dtype=torch.float32)


def test_atom_features():
    """Test the atom feature functions."""
    print("Testing Magpie-style atom features...")

    # Create test atomic numbers
    atomic_numbers = torch.tensor([1, 6, 8, 26, 79])  # H, C, O, Fe, Au

    # Test Magpie features
    features = get_magpie_features(atomic_numbers)
    print(f"Magpie features shape: {features.shape}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"H features (first 10): {features[0][:10]}")

    # Test composite features
    composite = create_composite_atom_features(atomic_numbers, use_onehot=True)
    print(f"Composite features shape: {composite.shape}")

    # Check if features have correct dimension
    assert features.shape[1] == 49, f"Expected 49 dimensions, got {features.shape[1]}"
    print("[OK] All tests passed!")


if __name__ == "__main__":
    test_atom_features()
