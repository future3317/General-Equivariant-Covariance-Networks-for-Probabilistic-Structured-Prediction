"""Probabilistic distribution losses for structured prediction."""

from gecn.distributions.base import StructuredDistributionLoss
from gecn.distributions.gaussian import GaussianNLL
from gecn.distributions.student_t import StudentTNLL
from gecn.distributions.robust_surrogate import RobustSurrogateLoss

__all__ = [
    "StructuredDistributionLoss",
    "GaussianNLL",
    "StudentTNLL",
    "RobustSurrogateLoss",
]
