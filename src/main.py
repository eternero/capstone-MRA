"""
...
"""
import os
from src.classes.track import TrackPipeline
from src.classes.essentia_models import (discogs_effnet_emb,
                                         timbre_effnet_model,
                                         acoustic_effnet_model,
                                         danceability_effnet_model,
                                         voice_instrumental_effnet_model
)

if __name__ == '__main__':
    print(os.cpu_count())
    AUDIO_PATH = "src/audio/dataset_flac"
    track_pipeline = TrackPipeline(AUDIO_PATH)

    essentia_models_dict =  {
                            # This is the embedding Model
                            discogs_effnet_emb: [ 
                                                # These are the ML Models
                                                timbre_effnet_model,
                                                acoustic_effnet_model,
                                                danceability_effnet_model,
                                                voice_instrumental_effnet_model
                                                ]
                            }
    track_list = track_pipeline.run_pipeline(essentia_models_dict)
    track_df   = track_pipeline.get_track_dataframe()
    track_df.to_csv('track_df_flac_03_09_25v2.csv', index=False)
