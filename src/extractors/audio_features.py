"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

from typing import TYPE_CHECKING, Union, List
import gc
import numpy as np
from typing import Any
from essentia.standard import MonoLoader
from src.utils.parallel import load_essentia_model
from src.classes.essentia_algos import EssentiaAlgo
from src.classes.essentia_containers import (EssentiaAlgorithmTask,
                                             EssentiaModelTask,
                                             HarmoF0Task,
                                             FeatureTask
                                            )

# Handle type checking stuff
if TYPE_CHECKING:
    from src.classes.track import Track


class FeatureExtractor:
    """
    This class is in charge of providing methods which facilitate the creation of a data pipeline
    for the audio features that we will be extracting in this project.
    """

    @staticmethod
    def handle_harmoF0(track_mono : np.ndarray,
                       harmo_task : HarmoF0Task) -> dict[str, Any]:

        # Unpack task attributes
        harmonic_f0 = harmo_task.algorithm
        device      = harmo_task.device

        # Return results!
        return harmonic_f0(track_mono, device)

    @staticmethod
    def retrieve_algorithm_features(track_mono : np.ndarray, track_path : str,
                                    essentia_algo_tasks : EssentiaAlgorithmTask) -> dict[str, Any]:
        """TODO : Add docstring"""

        algo_features  = {}
        algo_task_list = essentia_algo_tasks.algorithms
        for algorithm in algo_task_list:

            # Special Case, since loudness requires stereo audio and it loads it inside the method.
            if algorithm is EssentiaAlgo.get_loudness_ebu_r128:
                algo_features.update(algorithm(track_path))

            else:
                algo_features.update(algorithm(track_mono))

        return algo_features


    @staticmethod
    def retrieve_model_features(track_mono           : np.ndarray,
                                essentia_model_tasks : EssentiaModelTask) -> dict[str, Any]:
        """TODO : Add docstring"""

        # Unpack task attributes
        essentia_embs    = essentia_model_tasks.embedding_model
        inference_models = essentia_model_tasks.inference_models

        # Load and Cache the Embedding Model
        embeddings_tf    = load_essentia_model(essentia_embs.algorithm,
                                               essentia_embs.graph_filename,
                                               essentia_embs.output)

        # Compute the embeddings
        track_embeddings = embeddings_tf(track_mono)
        model_features   = {}  # Used to save our features!

        for essentia_inf_model in inference_models:
            inference_tf = load_essentia_model(essentia_inf_model.algorithm,
                                               essentia_inf_model.graph_filename,
                                               essentia_inf_model.output)

            # Gather Inference Predictions and save it as a feature.
            predictions  = inference_tf(track_embeddings)

            feat_index   = essentia_inf_model.target_index
            feature_name = [essentia_inf_model.classifiers[feat_index],
                            essentia_inf_model.model_family]
            feature_name = '_'.join(feature_name)
            model_features[feature_name] = np.mean(predictions, axis=0)[feat_index]

        return model_features


    @staticmethod
    def retrieve_all_essentia_features(track_path         : str,
                                       essentia_task_list : List[FeatureTask]):
        """TODO : Add docstring"""

        # Placeholders.
        track_mono_16 = MonoLoader(filename = track_path, sampleRate = 16000, resampleQuality = 0)()
        track_mono_44 = MonoLoader(filename = track_path, sampleRate = 44100, resampleQuality = 0)()
        curr_features = None
        all_features  = {}

        # Loop through the embedding -> model list in dictionary
        for essentia_obj in essentia_task_list:

            # Both Essentia Models and HarmoF0 use a Sample Rate of 16kHz
            if isinstance(essentia_obj, (EssentiaModelTask, HarmoF0Task)):

                # Check whether its an Essentia Model or HarmoF0, since they're handled differently
                if isinstance(essentia_obj, EssentiaModelTask):
                    curr_features = FeatureExtractor.retrieve_model_features(track_mono_16, essentia_obj)

                # If it isn't an Essentia Model, then it must be HarmoF0.
                else:
                    FeatureExtractor.handle_harmoF0(track_mono_16, essentia_obj)

            # If it is neither of those, then it must be an Essentia Algoritm, which uses sr = 44kHz
            else:
                curr_features = FeatureExtractor.retrieve_algorithm_features(track_mono=track_mono_44,
                                                                             track_path=track_path,
                                                                             essentia_algo_tasks=essentia_obj
                                                                            )

            # At the end of every iteration, which handles an Essentia Task, update our results!
            all_features.update(curr_features)

        # Clean up large audio arrays
        del track_mono_16
        del track_mono_44
        gc.collect() # Hint to Python to release memory

        return all_features
