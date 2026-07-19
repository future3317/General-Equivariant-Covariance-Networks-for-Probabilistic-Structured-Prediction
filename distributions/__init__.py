"""Probabilistic distribution losses for structured prediction."""

from distributions.base import StructuredDistributionLoss
from distributions.gaussian import GaussianNLL
from distributions.student_t import StudentTNLL
from distributions.robust_surrogate import RobustSurrogateLoss

__all__ = [
    "StructuredDistributionLoss",
    "GaussianNLL",
    "StudentTNLL",
    "RobustSurrogateLoss",
]
