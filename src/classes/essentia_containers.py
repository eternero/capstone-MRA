"""
This file aims to create dataclasses which will be used to efficiently handle the input to our
Track Pipeline. This is done based on the assumption that we will use only one of three objects:
    (1) EssentiaModel (2) EssentiaAlgo (3) HarmoF0

NOTE : Join essentia_algos.py and essentia_models.py into essentia_interfaces.py
"""

from dataclasses import dataclass
from typing import List, Callable, Union
from src.classes.essentia_algos import EssentiaAlgo
from src.classes.essentia_models import EssentiaModel

# -------------------------------------------------------------------------------------------------
# Define the Task Container Dataclasses
# -------------------------------------------------------------------------------------------------

@dataclass
class EssentiaModelTask:
    """Container for Essentia model-based feature extraction."""
    embedding_model  : EssentiaModel
    inference_models : List[EssentiaModel]

@dataclass
class EssentiaAlgorithmTask:
    """Container for Essentia algorithm-based feature extraction."""
    algorithms: List[Callable]

@dataclass
class HarmoF0Task:
    """Container for HarmoF0 pitch feature extraction."""
    algorithm : Callable = EssentiaAlgo.harmonic_f0
    device    : str      = 'mps'

# For type checking in whatever uses this!
FeatureTask = Union[EssentiaModelTask, EssentiaAlgorithmTask, HarmoF0Task]
