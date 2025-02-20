"""..."""
import os
from src.classes.track import TrackPipeline
from src.classes.essentia_models import EssentiaModel
from src.extractors.audio_features import FeatureExtractor

from concurrent.futures import ProcessPoolExecutor


if __name__ == '__main__':

    # First, define the embedding models and inference models to be used.
    # I'll only be using 1:1 in this example for the sake of simplicity.

    # ------------------------------------------------------------------------------
    # Discog Embedding Model
    # ------------------------------------------------------------------------------
    discogs_effnet_emb = EssentiaModel(
        graph_filename='src/embeddings/discogs-effnet-bs64-1.pb',
        output='PartitionedCall:1',
        algorithm='TensorflowPredictEffnetDiscogs',
    )


    # ------------------------------------------------------------------------------
    # Discog Inference Model
    # ------------------------------------------------------------------------------
    timbre_effnet_model = EssentiaModel(
        graph_filename='src/models/timbre-discogs-effnet-1.pb',
        output='model/Softmax',
        algorithm='TensorflowPredict2D',
    )


    # Now, it might be appropiate to extract a couple of tracks!
    AUDIO_PATH = "src/audio/dataset_flac"
    track_pipeline = TrackPipeline(AUDIO_PATH)
    track_filenames = os.listdir(AUDIO_PATH)

    track_list = track_pipeline.run_in_parallel(track_pipeline.load_single_track, track_filenames)
    track_list = [track for track in track_list if track is not None][:20] # Reduce to a max of 20.


    with ProcessPoolExecutor(max_workers = 10) as executor:
        futures = [
            executor.submit(FeatureExtractor.retrieve_model_features, track,
                            discogs_effnet_emb,
                            timbre_effnet_model
                           )
            for track in track_list
        ]
