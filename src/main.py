"""..."""
import os
from src.classes.track import TrackPipeline
from src.classes.essentia_models import (# Models
                                         danceability_effnet_model,
                                         mood_aggressive_effnet_model,
                                         mood_happy_effnet_model,
                                         mood_party_effnet_model,
                                         mood_relaxed_effnet_model,
                                         mood_sad_effnet_model,
                                         mood_acoustic_effnet_model,
                                         mood_electronic_effnet_model,
                                         voice_instrumental_effnet_model,
                                         voice_gender_effnet_model,
                                         tonal_atonal_effnet_model,
                                         timbre_effnet_model,
                                         nsynth_timbre_effnet_model,
                                         # Approach Models
                                         approachability_2c,
                                        #  approachability_3c,
                                         approachability_regression,
                                         # Engage Models
                                         engagement_2c,
                                        #  engagement_3c,
                                         engagement_regression,
                                         # Embeddings
                                         discogs_effnet_emb
                                        )
from src.classes.essentia_algos import EssentiaAlgo

if __name__ == '__main__':
    print(os.cpu_count())
    AUDIO_PATH = "src/audio/dataset_flac"
    track_pipeline = TrackPipeline(AUDIO_PATH)

    essentia_models_dict =  {
                            # This is the embedding Model
                            discogs_effnet_emb: [
                                                # These are the ML Models
                                                danceability_effnet_model,
                                                mood_aggressive_effnet_model,
                                                mood_happy_effnet_model,
                                                mood_party_effnet_model,
                                                mood_relaxed_effnet_model,
                                                mood_sad_effnet_model,
                                                mood_acoustic_effnet_model,
                                                mood_electronic_effnet_model,
                                                voice_instrumental_effnet_model,
                                                tonal_atonal_effnet_model,
                                                timbre_effnet_model,
                                                nsynth_timbre_effnet_model,
                                                ]
                            }

    # NOTE : Removed intensity and time signature given that they're not very accurate.
    essentia_objs_dict  =  {
                           "algorithms" : [
                                          EssentiaAlgo.el_monstruo,
                                          EssentiaAlgo.get_bpm_re2013,
                                          EssentiaAlgo.get_energy,
                                          EssentiaAlgo.get_loudness_ebu_r128
                                          ],
                            discogs_effnet_emb: [
                                                # These are the ML Models
                                                danceability_effnet_model,
                                                mood_aggressive_effnet_model,
                                                mood_happy_effnet_model,
                                                mood_party_effnet_model,
                                                mood_relaxed_effnet_model,
                                                mood_sad_effnet_model,
                                                mood_acoustic_effnet_model,
                                                mood_electronic_effnet_model,
                                                voice_instrumental_effnet_model,
                                                voice_gender_effnet_model,
                                                tonal_atonal_effnet_model,
                                                timbre_effnet_model,
                                                nsynth_timbre_effnet_model,
                                                approachability_2c,
                                                approachability_regression,
                                                engagement_2c,
                                                engagement_regression
                                                # Excluded 3c models since I gotta make some changes first.
                                                ]
                           }

    track_list = track_pipeline.run_pipeline(essentia_objs_dict)
    track_df   = track_pipeline.get_track_dataframe()
    track_df.to_csv('03_29_25_test_features_full.csv', index=False)
