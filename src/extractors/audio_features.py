"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

from functools import lru_cache
from typing import TYPE_CHECKING
import librosa
import numpy as np
import essentia.standard as es
from essentia.standard import MonoLoader
from src.classes.essentia_models import EssentiaModel

# Import Track only for type-checking; this import will not be executed at runtime.
if TYPE_CHECKING:
    from src.classes.track import Track


@lru_cache(None)
def load_essentia_model(algorithm_name : str, graph_filename : str, output_name : str):
    """..."""
    model_callable = getattr(es, algorithm_name)
    model_tf       = model_callable(graphFilename = graph_filename,
                                    output        = output_name)
    return model_tf


class FeatureExtractor:
    """
    This class is in charge of providing methods which facilitate the creation of a data pipeline
    for the audio features that we will be extracting in this project.
    """
    @staticmethod
    def retrieve_all_model_features(track      : "Track",
                                    model_dict : dict[EssentiaModel, list[EssentiaModel]]) -> "Track":

        # Load the track mono so that we can process it
        track.track_mono = MonoLoader(filename        = track.track_path,
                                      sampleRate      = 44100, # Hardcoded for now, since
                                      resampleQuality = 0)()   # all our tracks are 44kHz

        # Loop through the embedding -> model list in dictionary
        for embedding_model, inf_model_list in model_dict.items():

            # Load and Cache the Embedding Model
            embeddings_tf = load_essentia_model(embedding_model.algorithm,
                                                embedding_model.graph_filename,
                                                embedding_model.output)

            # Compute the embeddings
            track_embeddings = embeddings_tf(track.track_mono)

            # Load each inference model in the list.
            for inf_model in inf_model_list:
                inference_tf = load_essentia_model(inf_model.algorithm,
                                                  inf_model.graph_filename,
                                                  inf_model.output)

                # Gather Inference Predictions and save it as a feature.
                predictions  = inference_tf(track_embeddings)
                feature_name = [inf_model.classifiers[0], inf_model.model_family]
                feature_name = '_'.join(feature_name)
                track.features[feature_name] = np.mean(predictions, axis=0)[0]

        return track


    @staticmethod
    def retrieve_bpm_librosa(track : "Track"):
        """
        ...
        """
        tempo, _ = librosa.beat.beat_track(y=track.track_mono, sr = 44100)
        print(track.track_path)
        print(f"Detected BPM: {tempo}")
        print("-"*100)
