"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

from typing import TYPE_CHECKING, Callable
import librosa
import numpy as np
from src.utils.parallel import load_essentia_model
from src.classes.essentia_models import EssentiaModel

# Import Track only for type-checking; this import will not be executed at runtime.
if TYPE_CHECKING:
    from src.classes.track import Track


class FeatureExtractor:
    """
    This class is in charge of providing methods which facilitate the creation of a data pipeline
    for the audio features that we will be extracting in this project.
    """

    @staticmethod
    def retrieve_algorithm_features(track : "Track", algorithm_list : list[Callable]) -> "Track":
        for algorithm in algorithm_list:
            algorithm(track)
        return track


    @staticmethod
    def retrieve_model_features(track : "Track", essentia_embeddings : EssentiaModel,
                                essentia_inference_model_list : list[EssentiaModel]):

        # Load and Cache the Embedding Model
        embeddings_tf    = load_essentia_model(essentia_embeddings.algorithm,
                                               essentia_embeddings.graph_filename,
                                               essentia_embeddings.output)

        # Compute the embeddings
        track_embeddings = embeddings_tf(track.track_mono_16)

        for essentia_inf_model in essentia_inference_model_list:
            inference_tf = load_essentia_model(essentia_inf_model.algorithm,
                                               essentia_inf_model.graph_filename,
                                               essentia_inf_model.output)

            # Gather Inference Predictions and save it as a feature.
            predictions  = inference_tf(track_embeddings)

            feat_index   = essentia_inf_model.target_index
            feature_name = [essentia_inf_model.classifiers[feat_index],
                            essentia_inf_model.model_family]
            feature_name = '_'.join(feature_name)
            track.features[feature_name] = np.mean(predictions, axis=0)[feat_index]

        return track


    @staticmethod
    def retrieve_all_essentia_features(track             : "Track",
                                    essentia_obj_dict : dict[EssentiaModel, list[EssentiaModel]]) -> "Track":

        # Load the track monos so that we can process the track.
        track.track_mono_16 = track.get_track_mono(sample_rate = 16000)
        track.track_mono_44 = track.get_track_mono(sample_rate = 44100)

        # Loop through the embedding -> model list in dictionary
        for essentia_obj_type, essentia_obj_list in essentia_obj_dict.items():

            # Check Essentia Object Type. It can be either an Algorithm or Embedding
            if essentia_obj_type == "algorithms":
                track = FeatureExtractor.retrieve_algorithm_features(track, essentia_obj_list)

            # If it not an algorithm, then it must be a model!
            else:
                track = FeatureExtractor.retrieve_model_features(track,
                                                                 essentia_embeddings=essentia_obj_type,
                                                                 essentia_inference_model_list=essentia_obj_list)

        return track


    @staticmethod
    def retrieve_bpm_librosa(track : "Track"):
        """
        ...
        """
        tempo, _ = librosa.beat.beat_track(y=track.track_mono_44, sr = 44100)
        print(track.track_path)
        print(f"Detected BPM: {tempo}")
        print("-"*100)
