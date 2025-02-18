"""
...
"""
import os
import time
import random
from typing import Any, List, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from requests.exceptions import HTTPError, Timeout

from dotenv import load_dotenv
from essentia.standard import MonoLoader
from src.classes.essentia_models import EssentiaModel
from src.extractors.metadata import MetadataExtractor
from src.extractors.audio_features import FeatureExtractor
from src.extractors.spotify_api import SpotifyAPI, request_access_token


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


class TrackPipline:
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

    @staticmethod
    def _load_single_track(track_filename: str, base_path: str, sample_rate: int) -> Track:
        """
        Helper method for `load_tracks()`. Loads a single track and returns a Track object.
    
        Args:
            track_filename (str): The filename of the track.
            base_path      (str): The directory where tracks are stored.
            sample_rate    (int): The sample rate for loading the track.
    
        Returns:
            Track: A Track object if successful, or None if loading fails.
        """
        full_path = os.path.join(base_path, track_filename)
        if os.path.isfile(full_path):
            try:
                track_mono = MonoLoader(filename=full_path,
                                        sampleRate=sample_rate,
                                        resampleQuality=0)()
                track = Track(track_path=full_path, track_mono=track_mono)
                print(f"Loaded track: {track_filename}")
                return track

            except Exception as e:
                print(f"Error loading track {track_filename}: {e}")
        return None

    def load_tracks(self, num_processes: int = None) -> None:
        """
        Loads all audio files from base_path using process-based parallelism.
    
        Args:
            num_processes (int, optional): Number of worker processes to use.
                                         Defaults to the number of available CPU cores.
        """
        if num_processes is None:
            num_processes = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None

        track_filenames = os.listdir(self.base_path)

        # Use ProcessPoolExecutor with multiple arguments by mapping over parallel iterables.
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(
                self._load_single_track,
                track_filenames,
                [self.base_path] * len(track_filenames),
                [self.sample_rate] * len(track_filenames)
            ))

        # Filter out any None results (failed loads)
        self.track_list = [track for track in results if track is not None]


    def _get_metadata_and_spotify(self, track : Track, access_token : str):
        """Helper method for `run_parallel_metadata_and_spotify`. Allows to acquire the metadata
        and Spotify API features for a single track,

        Args:
            track      (Track): The track for which we will be acquiring its metadata and features.
            access_token (str): The access token needed to make the Spotify API requests.
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


    def run_parallel_metadata_and_spotify(self, access_token: str,
                                          num_workers: int = os.cpu_count()) -> None:
        """
        Runs the _get_metadata_and_spotify method in parallel for all tracks in self.track_list.
        
        Args:
            access_token (str): Spotify API access token.
            num_workers (int, optional): Number of parallel worker threads. Defaults to
                the minimum of available CPU cores and number of tracks.
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit each track for processing.
            futures = [
                        executor.submit(self._get_metadata_and_spotify, track, access_token)
                        for track in self.track_list
                      ]

            # Optionally, wait for each future to complete and handle exceptions.
            for future in as_completed(futures):
                try:
                    future.result()

                except Exception as e:
                    print(f"Error processing a track: {e}")


    def run_pipeline(self,
                     essentia_models_dict : Dict[EssentiaModel, List[EssentiaModel]]
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
            self.load_tracks()

        # -----------------------------------------------------------------------------------------
        # Step 1 : Metadata Extraction and Spotify API Features Extraction (Parallel)
        # -----------------------------------------------------------------------------------------
        load_dotenv()
        client_id     = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        access_token  = request_access_token(client_id, client_secret)
        self.run_parallel_metadata_and_spotify(access_token)


        # -----------------------------------------------------------------------------------------
        # Step 2 : Essentia Models Extraction (TODO : Needs to be concurrent)
        # -----------------------------------------------------------------------------------------
        for essentia_embeddings, essentia_model_list in essentia_models_dict.items():
            # First, gather the embeddings so that we only call them once.
            # NOTE : I'm seeing that it might be useless to have the `embeddings`
            #        attribute for this class... fix that!!! TODO TODO TODO
            essentia_embeddings = essentia_embeddings.get_model()

            for essentia_model in essentia_model_list:
                # Now, iterate through each of the models that pertain to a set embedding model
                # and use it to retrieve a feature from our tracks! Similar to the embeddings,
                # the model will only be loaded once and ran for each track. Efficiency!
                inference_model = essentia_model.get_model()
                feature_name    = essentia_model.get_graph_filename()

                for track in self.track_list:
                    self.audio_feature_extractor.retrieve_model_features(track,
                                                                         essentia_embeddings,
                                                                         inference_model,
                                                                         feature_name)

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