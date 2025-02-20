"""
...
"""
import os
import time
import random
from typing import Any, List, Dict, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from essentia.standard import MonoLoader
from requests.exceptions import HTTPError, Timeout
from src.extractors.metadata import MetadataExtractor
from src.classes.essentia_models import EssentiaModel
from src.extractors.audio_features import FeatureExtractor
from src.extractors.spotify_api import SpotifyAPI, request_access_token


import numpy as np

# Global variables to hold the models and feature name in each worker process.
global_embedding_model = None
global_inference_model = None
global_feature_name = None

def init_essentia_worker(embedding_model_callable, inference_model_callable, feature_name: str):
    """
    Initializer for each worker process. It loads the Essentia models into global variables.
    
    Args:
        embedding_model_callable (Callable): A callable that returns the embedding model.
        inference_model_callable (Callable): A callable that returns the inference model.
        feature_name (str): The name of the feature to extract.
    """
    global global_embedding_model, global_inference_model, global_feature_name
    global_embedding_model = embedding_model_callable()  # Load the embedding model
    global_inference_model = inference_model_callable()  # Load the inference model
    global_feature_name = feature_name

def retrieve_features_worker(track):
    """
    Worker function to extract audio features for a single track using the global Essentia models.
    
    Args:
        track (Track): A Track object.
        
    Returns:
        The Track object with the extracted feature added.
    """
    # Compute track embeddings using the global embedding model.
    track_embeddings = global_embedding_model(track.track_mono)
    # Get model predictions using the global inference model.
    model_predictions = global_inference_model(track_embeddings)
    # Compute the mean prediction and store it in the track's features.
    track.features[global_feature_name] = np.mean(model_predictions, axis=0)
    return track


class Track:
    """
    The Track object should encapsulate all of the necessary elements that we've determined to be 
    pertinent to describing and analyzing a song. This includes metadata, features extracted using
    essentia, and other information acquired from the SpotifyAPI (TODO).

    This class implements the usage of others such as the MetadataExtractor, SpotifyAPI, 
    EssentiaModels and FeatureExtractor. The last two essentially work towards the same thing,
    acquisition of features through essentia - and the first two are quite self-explanatory.

    Attributes:
        track_path : The path which points to the location of the song file.
        track_mono : The MonoLoader (or numpy array) which represents the song.

        features   : The dictionary which stores the features that have been acquired for the track
        metadata   : The metadata of the track. Includes track name, album name, artist name etc...
        metadata_extractor : The instance of `MetadataExtractor` used to acquire the metadata.

    TODO : Missing implementation of SpotifyAPI here, work on it.
    """
    def __init__(self, track_path : str, track_mono : MonoLoader):
        self.track_path = track_path
        self.track_mono = track_mono

        self.features: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def update_features(self, new_features: Dict[str, Any]) -> None:
        """_summary_

        Args:
            new_features (Dict[str, Any]): _description_
        """
        self.features.update(new_features)

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """_summary_

        Args:
            new_metadata (Dict[str, Any]): _description_
        """
        self.metadata = new_metadata


class TrackPipeline:
    """
    A data pipeline that processes audio tracks in a prescribed order:
        1. Metadata Extraction 
        2. Spotify API Extraction
        3. Additional Tag Extraction (e.g. Essentia Models / Algorithms, Librosa, TorchAudio)
    """

    def __init__(self, base_path : str, sample_rate: int = 44100):
        self.base_path               = base_path
        self.sample_rate             = sample_rate
        self.track_list: List[Track] = []

        # Define extractors to help ourselves.
        self.spotify_api_extractor   = SpotifyAPI()
        self.metadata_extractor      = MetadataExtractor()
        self.audio_feature_extractor = FeatureExtractor()

        # NOTE : Make it so that the `FeatureExtractor` class works similar
        # to these previous two which work on a single track, rather than a list.

    def load_single_track(self, track_filename: str) -> Track:
        """
        Loads a single track and returns a Track object.
    
        Args:
            track_filename : The filename of the track.
    
        Returns:
            Track: A Track object if successful, or None if loading fails.
        """
        full_path = os.path.join(self.base_path, track_filename)
        if os.path.isfile(full_path):
            try:
                track_mono = MonoLoader(filename       =full_path,
                                        sampleRate     =self.sample_rate,
                                        resampleQuality=0)()

                track = Track(track_path=full_path, track_mono=track_mono)
                print(f"Loaded track: {track_filename}")
                return track

            except Exception as e:
                print(f"Error loading track {track_filename}: {e}")
        return None


    def _get_metadata_and_spotify(self, track : Track, access_token : str):
        """Acquires the metadata and Spotify API features for a single track.

        Args:
            track        : The track for which we will be acquiring its metadata and features.
            access_token : The access token needed to make the Spotify API requests.
        """
        metadata = self.metadata_extractor.extract(track.track_path)
        track.update_metadata(metadata)


        # Once we've got the metadata, we can proceed to get the Spotify API Features
        track_name       = track.metadata['TITLE']
        track_artist     = track.metadata['ARTIST']
        print(f"Current : {track_artist} : {track_name}")

        # Handle errors using a timeout with exponential backoff. Let us define our params first.
        retry_limit = 5     # max number of retries to attempt.
        base_delay  = 1     # initial delay in seconds.
        retries     = 0     # used to track number of retries.


        while True:
            try:
                spotify_features = self.spotify_api_extractor.get_spotify_features(track_artist,
                                                                                track_name,
                                                                                access_token)
                track.update_features(spotify_features)
                break   # Upon success, exit the loop.

            # If we catch an error, we retry and delay our calls until we find success.
            except (HTTPError, Timeout, ConnectionError) as err:
                print(f"Error retrieving Spotify features for {track_artist} - {track_name}: {err}")

                retries += 1
                if retries > retry_limit:
                    print("Max retries reached. Skipping track.")
                    spotify_features = {}  # or handle as needed
                    break

                # Calculate delay with exponential backoff and some jitter
                delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)


    def run_in_parallel(self, func   : Callable,
                        item_list    : list[Any], *args,
                        num_workers  : int = os.cpu_count(),
                        executor_type: str = "process",
                        **kwargs)   -> list[Any]:
        """
        Runs the provided function in parallel for each item in item_list. This is essentially
        a wrapper of the concurrent.futures XPoolExecutor methods adapted to our use case.

        Args:
            func          : The method to be run in parallel.
            item_list     : Will either be a list of all track filenames, or a list of all tracks.
            num_workers   : Number of workers to use for our executor.
            executor_type : The executor to be used, either ThreadPoolExec or ProcessPoolExec.
        """

        # First, determine which Executor will be used.
        if executor_type.lower() == "process":
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor

        results = []    # List to collect all results later on.
        with Executor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(func, item, *args, **kwargs)
                for item in item_list
            ]

            # Wait for all tasks to complete and handle exceptions.
            for future in as_completed(futures):
                try:
                    result = future.result()    # This is done to handle
                    results.append(result)      # all results asynchronously
                                                # and individually.
                except Exception as e:
                    print(f"Error processing a track: {e}")

        return results


    def run_parallel_feature_extraction(self,
                                        embedding_model_obj: Any,
                                        inference_model_obj: Any,
                                        feature_name: str,
                                        num_workers: int = os.cpu_count()) -> None:
        """
        Parallelizes audio feature extraction using ProcessPoolExecutor with an initializer.
    
        Args:
            embedding_model_obj: An object with a get_model() method to load the embedding model.
            inference_model_obj: An object with a get_model() method to load the inference model.
            feature_name (str): The name of the feature being extracted.
            num_workers (int, optional): Number of worker processes.
        """
        # Use ProcessPoolExecutor with initializer to load the heavy models in each worker.
        with ProcessPoolExecutor(max_workers=num_workers,
                                  initializer=init_essentia_worker,
                                  initargs=(embedding_model_obj.get_model,
                                            inference_model_obj.get_model,
                                            feature_name)
                                 ) as executor:
            futures = [
                executor.submit(retrieve_features_worker, track)
                for track in self.track_list
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing a track in feature extraction: {e}")


    def run_pipeline(self, essentia_models_dict : Dict[EssentiaModel, List[EssentiaModel]]
                    ) -> List[Track]:
        """
        Processes all tracks in the following order:
          1. Extract metadata
          2. Extract Spotify features (requires metadata)
          3. Run any additional steps (Essentia Models, Essentia Algorithms, etc.)

        Args:
            essentia_models_dict : This is a dictionary that contains pairs of 
                        `{Essentia Embeddings : List[Essentia Model]}`
                 
                This is because multiple models can depend on the same embeddings, 
                and containing them as such provides an efficient approach..
        """
        if not self.track_list:
            track_filenames = os.listdir(self.base_path)
            result_tracks   = self.run_in_parallel(self.load_single_track,
                                                   track_filenames,
                                                   executor_type="process")

            # Filter the results to only maintain retrieved tracks.
            self.track_list = [track for track in result_tracks if track is not None]
            print(f"Retrieved {len(self.track_list)} tracks.")

            # Reduce to 20 for the sake of testing
            self.track_list = self.track_list[:20]

        # -----------------------------------------------------------------------------------------
        # Step 1 : Metadata Extraction and Spotify API Features Extraction (Parallel)
        # -----------------------------------------------------------------------------------------
        load_dotenv()
        client_id     = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        access_token  = request_access_token(client_id, client_secret)

        self.run_in_parallel(self._get_metadata_and_spotify,
                             self.track_list, access_token,
                             executor_type="thread"
                            )

        # -----------------------------------------------------------------------------------------
        # Step 2 : Essentia Models Extraction (TODO : Needs to be concurrent)
        # -----------------------------------------------------------------------------------------
        for essentia_embeddings_obj, essentia_model_list in essentia_models_dict.items():

            for inference_model_obj in essentia_model_list:
                feature_name = inference_model_obj.get_graph_filename()
                print(f"CURRENT MODEL: {feature_name}")
                self.run_parallel_feature_extraction(essentia_embeddings_obj,
                                                     inference_model_obj,
                                                     feature_name,
                                                    )


        # -----------------------------------------------------------------------------------------
        # Step 3 : Essentia Algorithms Extraction
        # -----------------------------------------------------------------------------------------

        # NOTE : I could go about the above in two ways... we hardcode the algorithms that we will
        #        be using for feature retrieval or we do something similar to the previous step...
        #
        #        Providing a list of methods that will be used should work as well. For example, we
        #        can have an `add_pipeline_step()` method to which we provide it callables such as
        #               FeatureExtractor.retrieve_bpm_re2013,
        #               FeatureExtractor.retrieve_bpm_librosa
        #
        #        We just have to standardize these methods so that they all receive the same input,
        #        a `Track` object representing the current track. This should work even with
        #        librosa since it also takes a MonoLoader for its function inputs.
        return self.track_list



# TODO : Method that turns all this shit into a Pandas DF



# NOTE I'm thinking of two ways of parallelizing the Essentia Model calls.
#
# As ever, I must be aware that the parallelizing or anything really, first requires for the
# embeddings to be loaded. That is done on the outside then, since we will only load embeddings
# once or twice.
#
#
# What I would like to avoid however, is to load models far too often. That I can easily avoid.
# So, what are the two ways I'm thinking about. Actually I think I've got three maybe... idk
#
#
# 1. Load a model and go straight to paralellization. So just get the inference model, then pass
#    the model as a function parameter and boom.
#
# 2. Load all models for an embedding (This action will be kept linear for simplicity).



"""
Thoughts on model loading.

What seems to be taking a large chunk of time here is the loading of Embedding Models and Inference
Models from Essentia... Of course, this is something that I would like to avoid. 

Preloading these models would be the ideal scenario as I've considered so far. An idea of this 
would be that I ge the pipeline running, and while it works on loading the tracks and then getting
the metadata the spotify API featuers, then another thread, fork, process, would be running in the
background loading up all the models.

I'm not exactly sure how I could implement this in Python In C I would just run a fork, a child
process at the start while all of the other actions are running... 


NOTE
Apparently loading doesn't take much... barely any time at all actually, I'd reckon it should take
less than 5 seconds for all models to be loaded. So I think that for the sake of readability, I won't
be pre-loading the models.
"""