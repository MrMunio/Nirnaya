"""Nirnaya (Telugu/Sanskrit: "decision") - single-pass typed decisions with calibrated confidence."""
from .calibration import Calibrator
from .engine import NirnayaEngine
from .llama_engine import LlamaCppEngine
from .model_registry import ModelRegistry, registry
from .prism_engine import PrismLlamaEngine
from .schema import Answer, Prediction

__all__ = [
    "NirnayaEngine",
    "PrismLlamaEngine",
    "LlamaCppEngine",
    "ModelRegistry",
    "registry",
    "Calibrator",
    "Answer",
    "Prediction",
]
__version__ = "0.1.0"

