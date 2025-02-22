"""
...
"""
import os
import time
import random
from typing import Any, List, Dict, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


import pandas as pd
from dotenv import load_dotenv
from essentia.standard import MonoLoader
from requests.exceptions import HTTPError, Timeout
from src.extractors.metadata import MetadataExtractor
from src.classes.essentia_models import EssentiaModel
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
            self.track_list = self.track_list[:100]

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
        for essentia_embeddings, essentia_model_list in essentia_models_dict.items():

            for essentia_model in essentia_model_list:

                start  = time.time()
                result = self.run_in_parallel(self.audio_feature_extractor.retrieve_model_features_v2,
                                     self.track_list,
                                     essentia_embeddings,       # First the embeddings
                                     essentia_model,            # then the inference model
                                     executor_type="process"
                                    )
                
                self.track_list = [track for track in result if track is not None]
                print(f"Executed in {time.time() - start}s")

        return self.track_list


    def get_track_dataframe(self) -> pd.DataFrame:
        """
        This method is to be used only once the `run_pipeline()` method has been run and 
        succesfully completed. It will compile all of the track features into one huge dataframe. 
        These features will only be acquired from the dictionary stored in `track.features`.

        Additionally, the tracks will be 'tagged' with the first four columns, which will include

                    FILENAME  |  ARTIST  |  TITLE  |  ALBUM

        These are all attributes which can be acquired from `track.metadata` as specified in the 
        column names (e.g. filename = track.get_metadata()['FILENAME']) and these MUST be the first
        couple of columns since they allow to easily identify the tracks when visually analyzing...

        NOTE : Feature Names found on `track.features` are not all predetermined due to the 
        the different Essentia Models that might be used. This means that one will likely have to
        first retrieve all of the `key` values from `track.features` before proceeding with the
        creation of the dataframe.

        Or perhaps, given that every single track in `self.track_list` will undoubtedly contain the
        same features, then the dataframe could be automatically done or something else could be 
        done i don't know...
        """
        rows = []
        # Define the required metadata columns
        required_columns = ["FILENAME", "ARTIST", "TITLE", "ALBUM"]

        for track in self.track_list:
            # Get metadata (using .get ensures missing keys return None)
            metadata = {col: track.metadata.get(col, None) for col in required_columns}
            # Merge with features (which may have arbitrary keys)
            row = {}
            row.update(metadata)
            row.update(track.features)
            rows.append(row)

        # Create DataFrame from list of row dictionaries.
        df = pd.DataFrame(rows)

        # # Determine all columns in the DataFrame.
        # all_columns = list(df.columns)
        # # Ensure required columns are first; maintain their order.
        # remaining_columns = [col for col in all_columns if col not in required_columns]
        # new_order = required_columns + remaining_columns
        # df = df[new_order]

        return df
