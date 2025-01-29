"""
NOTE This code is still in development, trying out different features. Nonetheless, if we
stick to essentia it might be best to join this with classes/essentia_models.py
"""

# Imports
import librosa
import numpy as np
import pandas as pd
import essentia.standard as es
from src.classes.track import Track
from src.classes.essentia_models import EssentiaModel


class FeatureExtractor:
    """
    This class is in charge of providing methods which facilitate the creation of a data pipeline
    for the audio features that we will be extracting in this project.
    """
    def __init__(self, track_list : list[Track]):
        self.track_list = track_list

    def retrieve_model_features(self,
                         emb_and_model_dict : dict[EssentiaModel,
                                              list[EssentiaModel]]) -> list[Track]:
        """
        ...
        """

        # First, iterate through the embeddings, acquiring the embedding model
        # for each of the essentia embeddings being used.
        for emb, model_list in emb_and_model_dict.items():
            embedding_model = emb.get_model()

            # The same process is repeated for our models.
            for model in model_list:
                inference_model = model.get_model()

                for track in self.track_list:
                    track_embeddings  = embedding_model(track.get_track_mono())
                    model_predictions = inference_model(track_embeddings)

                    # We should make sure to compress the model_predictions into a shape of (2, )
                    # e.g. [x, y] so that it can actually be interpreted in our data.
                    mean_predictions = np.mean(model_predictions, axis=0)

                    curr_feat = model.get_graph_filename()
                    track.features[curr_feat] = mean_predictions


        # Ideally, we will return a modified version of the track list, in which all of the tracks
        # contain their newly provided tags.
        return self.track_list

    def retrieve_bpm_re2013(self):
        """
        ...
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

        for track in self.track_list:
            bpm, beats, beats_confidence, _, _ = rhythm_extractor(track.get_track_mono())
            print(track.get_track_path())
            print("BPM:", bpm)
            print("Beat positions (sec.):", beats)
            print("Beat estimation confidence:", beats_confidence)
            print("-"*100)

    def retrieve_bpm_librosa(self):
        """
        ...
        """
        for track in self.track_list:
            tempo, _ = librosa.beat.beat_track(y=track.get_track_mono(), sr = 44100)
            print(track.get_track_path())
            print(f"Detected BPM: {tempo}")
            print("-"*100)


    def create_dataframe(self) -> pd.DataFrame:
        """
        Self explanatory.

        Have to run previous function and then this one! Perhaps I could then set it up to do 
        some method chaining.
        """
        dataframe_rows = []     # Variable to gather dataframe rows.
        for track in self.track_list:

            # Gather the track name and features
            curr_row = {'track_path' : track.get_track_path()}
            curr_row.update(track.get_features())

            # Append to our list of rows
            dataframe_rows.append(curr_row)

        return pd.DataFrame(dataframe_rows)
