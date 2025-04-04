"""..."""
from src.classes.track import TrackPipeline
from src.classes.essentia_algos import EssentiaAlgo
from src.classes.essentia_containers import (EssentiaAlgorithmTask,
                                             EssentiaModelTask,
                                             HarmoF0Task
                                            )
import src.classes.essentia_models as essentia_models

if __name__ == '__main__':
    AUDIO_PATH = "src/audio/testing_dataset_flac"
    track_pipeline = TrackPipeline(AUDIO_PATH)

    # Create our Tasks before running it...
    essentia_discogs_effnet_embeddings = essentia_models.discogs_effnet_emb
    essentia_discogs_effnet_models     = [
                                        essentia_models.danceability_effnet_model,
                                        essentia_models.mood_aggressive_effnet_model,
                                        essentia_models.mood_happy_effnet_model,
                                        essentia_models.mood_party_effnet_model,
                                        essentia_models.mood_relaxed_effnet_model,
                                        essentia_models.mood_sad_effnet_model,
                                        essentia_models.mood_acoustic_effnet_model,
                                        essentia_models.mood_electronic_effnet_model,
                                        essentia_models.voice_instrumental_effnet_model,
                                        essentia_models.voice_gender_effnet_model,
                                        essentia_models.tonal_atonal_effnet_model,
                                        essentia_models.timbre_effnet_model,
                                        essentia_models.nsynth_timbre_effnet_model,
                                        essentia_models.approachability_2c,
                                        essentia_models.approachability_regression,
                                        essentia_models.engagement_2c,
                                        essentia_models.engagement_regression
                                        ]
    essentia_discogs_effnet_task       = EssentiaModelTask(embedding_model =essentia_discogs_effnet_embeddings,
                                                           inference_models=essentia_discogs_effnet_models)

    essentia_algorithms_task           = EssentiaAlgorithmTask(algorithms=[
                                                                EssentiaAlgo.harmonic_f0,
                                                                EssentiaAlgo.el_monstruo,
                                                                EssentiaAlgo.get_bpm_re2013,
                                                                EssentiaAlgo.get_energy,
                                                                # EssentiaAlgo.get_loudness_ebu_r128
                                                               ])

    harmof0_task                       = HarmoF0Task()
    essentia_task_list                 = [essentia_algorithms_task, harmof0_task]

    track_list = track_pipeline.run_pipeline(essentia_task_list=essentia_task_list)
    track_df   = track_pipeline.get_track_dataframe()
    track_df.to_csv('datasets/04_04_25_refactored_dataset.csv', index=False)

