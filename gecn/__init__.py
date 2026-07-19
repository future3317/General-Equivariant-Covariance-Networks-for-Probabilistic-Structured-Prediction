"""General Equivariant Covariance Networks (gecn).

A modular library for building equivariant probabilistic predictors on
finite-dimensional orthogonal representations.
"""

__version__ = "0.1.0"

from gecn.representations import (
    OrthogonalRepresentationSpec,
    SymmetricSquareSpec,
    O3IrrepsSpec,
    O3SymmetricOperatorBasis,
    symmetric_square_irreps,
    rank2_symmetric_irreps,
    rank4_elasticity_irreps,
)
from gecn.spd_maps import (
    SPDMap,
    MatrixExponentialMap,
    SpectralSoftplusMap,
    SquarePlusIdentityMap,
    PrecisionExponentialMap,
    LowRankPlusIsotropicMap,
)
from gecn.distributions import (
    StructuredDistributionLoss,
    GaussianNLL,
    StudentTNLL,
    RobustSurrogateLoss,
)
from gecn.models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantSymmetricOperatorHead,
    O3EquivariantLowRankCovarianceHead,
    StructuredProbabilisticPredictor,
)

__all__ = [
    "OrthogonalRepresentationSpec",
    "SymmetricSquareSpec",
    "O3IrrepsSpec",
    "O3SymmetricOperatorBasis",
    "symmetric_square_irreps",
    "rank2_symmetric_irreps",
    "rank4_elasticity_irreps",
    "SPDMap",
    "MatrixExponentialMap",
    "SpectralSoftplusMap",
    "SquarePlusIdentityMap",
    "PrecisionExponentialMap",
    "LowRankPlusIsotropicMap",
    "StructuredDistributionLoss",
    "GaussianNLL",
    "StudentTNLL",
    "RobustSurrogateLoss",
    "EquivariantBackbone",
    "EquivariantMeanHead",
    "O3EquivariantSymmetricOperatorHead",
    "O3EquivariantLowRankCovarianceHead",
    "StructuredProbabilisticPredictor",
]
