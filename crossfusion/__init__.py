"""CrossFusion: Annotation-Driven Asymmetric Cross-Modal Attention."""
from .model import CrossFusion, AsymmetricCrossModalAttention, LearnedGating
from .loss import ConfidenceWeightedLoss
from .dataset import MultimodalSentimentDataset, build_dataloader

__version__ = "1.0.0"
__all__ = [
    "CrossFusion",
    "AsymmetricCrossModalAttention",
    "LearnedGating",
    "ConfidenceWeightedLoss",
    "MultimodalSentimentDataset",
    "build_dataloader",
]
