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
import src.classes.essentia_models as essentia_models

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

# -------------------------------------------------------------------------------------------------
# Define all the stuff will be used in the final stage of the project... constants.
# -------------------------------------------------------------------------------------------------
essentia_discogs_models = [
    essentia_models.danceability_effnet_model,
    essentia_models.mood_aggressive_effnet_model,
    essentia_models.mood_happy_effnet_model,
    essentia_models.mood_party_effnet_model,
    essentia_models.mood_relaxed_effnet_model,
    essentia_models.mood_sad_effnet_model,
    essentia_models.mood_acoustic_effnet_model,
    essentia_models.mood_electronic_effnet_model,
    essentia_models.voice_instrumental_effnet_model,
    essentia_models.voice_gender_effnet_model,
]

essentia_discogs_task    = EssentiaModelTask(embedding_model  = essentia_models.discogs_effnet_emb,
                                             inference_models = essentia_discogs_models)
essentia_algorithms_task = EssentiaAlgorithmTask(algorithms=[
                                                    EssentiaAlgo.mfcc_renewed,
                                                    EssentiaAlgo.get_bpm_re2013,
                                                    ])
essentia_task_list       = [essentia_discogs_task, essentia_algorithms_task]
