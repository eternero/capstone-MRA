"""..."""
from src.classes.track import TrackPipeline
from src.classes.essentia_algos import EssentiaAlgo
import src.classes.essentia_models as essentia_models
from src.classes.essentia_containers import EssentiaAlgorithmTask, EssentiaModelTask

if __name__ == '__main__':
    AUDIO_PATH = "src/audio/dataset_flac_2"
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
                                                                EssentiaAlgo.el_monstruo,
                                                                EssentiaAlgo.get_bpm_re2013,
                                                                EssentiaAlgo.get_energy,
                                                                EssentiaAlgo.get_loudness_ebu_r128
                                                               ])

    additional_tasks                   = EssentiaAlgo.harmonic_f0
    essentia_task_list                 = [essentia_algorithms_task, essentia_discogs_effnet_task]

    track_list = track_pipeline.run_pipeline(essentia_task_list = essentia_task_list,
                                             additional_tasks   = additional_tasks)



