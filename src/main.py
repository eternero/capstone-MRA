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
from src.classes.essentia_algos import EssentiaAlgo

if __name__ == '__main__':
    print(os.cpu_count())
    AUDIO_PATH = "src/audio/testing_dataset_flac"
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

    essentia_algos_dict  = {
                           "algorithms" : [
                                          EssentiaAlgo.el_monstruo,
                                          EssentiaAlgo.get_bpm_re2013,
                                          EssentiaAlgo.get_energy,
                                          EssentiaAlgo.get_time_signature,
                                          EssentiaAlgo.get_loudness_ebu_r128,
                                          EssentiaAlgo.get_intensity
                                          ],
                            discogs_effnet_emb: [
                                                # These are the ML Models
                                                timbre_effnet_model,
                                                acoustic_effnet_model,
                                                danceability_effnet_model,
                                                voice_instrumental_effnet_model
                                                ]
                           }

    track_list = track_pipeline.run_pipeline(essentia_algos_dict)
    track_df   = track_pipeline.get_track_dataframe()
    track_df.to_csv('metadata3.csv', index=False)
