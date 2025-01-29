"""
...
"""

# Imports
from src.classes.track import get_tracks
from src.classes.essentia_models import *    # import all so that we can also have access
                                             # to the embeddings and models that were defined.
from src.extractors.audio_features import FeatureExtractor


# First, acquire the tracks which we will evaluate
track_list = get_tracks('src/audio')


# Now, create the dictionary which includes the embeddings
# and the respective models which we will be using...
emb_and_model_dict = {
    discogs_effnet_emb : [timbre_effnet_model],

    msd_musicnn_emb : [danceability_musicnn_model,
                       voice_instrumental_musicnn_model,
                       mood_happy_musicnn_model,
                       mood_aggressive_musicnn_model]
}


feature_extractor = FeatureExtractor(track_list=track_list)
feature_extractor.retrieve_bpm_librosa()


# updated_track_list = feature_extractor.retrieve_model_features(emb_and_model_dict=emb_and_model_dict)
# features_df = feature_extractor.create_dataframe()
# features_df.to_csv('features.csv', index=False)
