"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

from typing import TYPE_CHECKING, List, Any
import gc
import numpy as np
from src.classes.essentia_algos import EssentiaAlgo
from src.utils.parallel import load_essentia_model, torch_load
from src.classes.essentia_containers import (EssentiaAlgorithmTask,
                                             EssentiaModelTask,
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
    def retrieve_algorithm_features(track               : "Track",
                                    essentia_algo_tasks : EssentiaAlgorithmTask) -> dict[str, Any]:
        """TODO : Add docstring"""

        algo_features  = {}
        algo_task_list = essentia_algo_tasks.algorithms
        for algorithm in algo_task_list:

            # Special Case, since loudness requires stereo audio and it loads it inside the method.
            if algorithm is EssentiaAlgo.get_loudness_ebu_r128:
                algo_features.update(algorithm(track.track_path))

            else:
                algo_features.update(algorithm(track.track_mono_44))

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

        # Compute and save the embeddings
        track_embeddings = embeddings_tf(track_mono)
        model_features   = {}  # Used to save our features + embeddings!
        emb_mean         = np.mean(track_embeddings, axis=0)
        model_features[essentia_embs.embedding_name] = list(emb_mean)

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
    def retrieve_all_essentia_features(track              : "Track",
                                       essentia_task_list : List[FeatureTask]):
        """TODO : Add docstring"""

        # Placeholders.
        track_mono_44, track_mono_16 = torch_load(track_path = track.track_path,
                                                  seg_start  = track.segment_start)

        track.track_mono_44 = track_mono_44
        track.track_mono_16 = track_mono_16

        # Loop through all the essentia tasks and handle them as necessary.
        for essentia_obj in essentia_task_list:

            # Check whether its an Essentia Model or Algorithm, since they're handled differently
            if isinstance(essentia_obj, EssentiaModelTask):
                curr_features = FeatureExtractor.retrieve_model_features(track.track_mono_16, essentia_obj)

            # If it is not a model, then it must be an Essentia Algoritm, which uses sr = 44kHz
            else:
                curr_features = FeatureExtractor.retrieve_algorithm_features(track             = track,
                                                                             essentia_algo_tasks=essentia_obj
                                                                            )

            # At the end of every iteration, which handles an Essentia Task, update our results!
            track.features.update(curr_features)


        del track.track_mono_44         # Clean up large audio arrays and hint for garbage collection
        gc.collect()                    # I kept track_mono_16 since it will be reused for Harmonic_F0.

        return track