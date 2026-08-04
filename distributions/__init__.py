"""Probabilistic distribution losses for structured prediction."""

from distributions.base import StructuredDistributionLoss
from distributions.gaussian import GaussianNLL
from distributions.robust_surrogate import RobustSurrogateLoss
from distributions.student_t import StudentTNLL

__all__ = [
    "GaussianNLL",
    "RobustSurrogateLoss",
    "StructuredDistributionLoss",
    "StudentTNLL",
]
