"""..."""
import os
import time
from src.classes.track import TrackPipeline
from src.utils.parallel import run_in_parallel
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
    AUDIO_PATH = "src/audio/dataset_mp3"
    track_pipeline = TrackPipeline(AUDIO_PATH)
    track_filenames = os.listdir(AUDIO_PATH)

    track_list = run_in_parallel(track_pipeline.load_single_track, track_filenames)
    track_list = [track for track in track_list if track is not None][:100] # Reduce to a max of 20.


    start = time.time()
    with ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:

        # Pre-load models here.
        # ... code to pre-load all models


        # Could be worked instead as a task-list, in which all tracks have
        # their list of models to be used to extract features.


        futures = [
            executor.submit(FeatureExtractor.retrieve_model_features_v2, track,
                            discogs_effnet_emb,     # Send the models
                            timbre_effnet_model     # down here.
                           )
            for track in track_list
        ]
    print(f"Executed in {time.time() - start}s")


# NOTE : Need to analyze how this shit currently works to then draw it up with multiple 
# models working for the same embeddings... as it should be.

# How would it work if we had 10 models per embedding model? 
# Should we pass in a list of models as a param and handle those
# in different processes?

# Or keep the relationship 1:1, and every time we change model we also
# pass in the embeddings... I don't think that's a big deal, it will be
# interesting to try out these two approaches.
