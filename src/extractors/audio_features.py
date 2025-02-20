"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

from functools import lru_cache
from typing import Callable, TYPE_CHECKING
import librosa
import numpy as np
import essentia.standard as es
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
    def retrieve_model_features(track : "Track",
                                embedding_model : EssentiaModel,
                                inference_model : EssentiaModel,
                                ):
        """
        ...
        """
        print(f"Extracting Features for {track.track_path}")

        embedding_callable = getattr(es, embedding_model.get_algorithm())
        inference_callable = getattr(es, inference_model.get_algorithm())

        embeddings_tf = embedding_callable(graphFilename = embedding_model.get_graph_filename(),
                                           output = embedding_model.get_output())

        inference_tf = inference_callable(graphFilename = inference_model.get_graph_filename(),
                                           output = inference_model.get_output())


        track_embeddings  = embeddings_tf(track.track_mono)
        model_predictions = inference_tf(track_embeddings)
        track.features[inference_model.get_graph_filename()] = np.mean(model_predictions, axis=0)

    @staticmethod
    def retrieve_bpm_re2013(track : "Track"):
        """
        ...
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(track.get_track_mono())
        print(track.get_track_path())
        print("BPM:", bpm)
        print("Beat positions (sec.):", beats)
        print("Beat estimation confidence:", beats_confidence)
        print("-"*100)

    @staticmethod
    def retrieve_bpm_librosa(track : "Track"):
        """
        ...
        """
        tempo, _ = librosa.beat.beat_track(y=track.get_track_mono(), sr = 44100)
        print(track.get_track_path())
        print(f"Detected BPM: {tempo}")
        print("-"*100)
