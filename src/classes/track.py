"""
...
"""
import os
import time
import random
from typing import Any, List, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import essentia
from essentia.standard import MonoLoader
from requests.exceptions import HTTPError, Timeout

from src.utils.parallel import run_in_parallel
from src.extractors.metadata import MetadataExtractor
from src.classes.essentia_models import EssentiaModel
from src.extractors.audio_features import FeatureExtractor
from src.extractors.spotify_api import SpotifyAPI, request_access_token

# DISABLE LOGGING. ANNOYING!
essentia.log.infoActive = False
essentia.log.warningActive = False


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
    def __init__(self, track_path : str):
        self.track_path    = track_path
        self.track_mono_16 = None
        self.track_mono_44 = None


        self.features      : Dict[str, Any] = {}
        self.metadata      : Dict[str, Any] = {}

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

    def get_track_mono(self, sample_rate : int) -> np.array:
        """Wrapper for the MonoLoader method."""
        return MonoLoader(filename        = self.track_path,
                          sampleRate      = sample_rate,
                          resampleQuality = 0)()



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

    def _get_metadata_and_spotify(self, track_path : Track, access_token : str):
        """Acquires the metadata and Spotify API features for a single track.

        Args:
            track_path   : The path to the track for which we will be acquiring its features.
            access_token : The access token needed to make the Spotify API requests.
        """
        # Initialize a `Track` object.
        track    = Track(track_path=track_path)
        metadata = self.metadata_extractor.extract(track.track_path)
        track.update_metadata(metadata)


        # Once we've got the metadata, we can proceed to get the Spotify API Features
        track_name       = track.metadata['TITLE']
        track_album      = track.metadata['ALBUM']
        track_artist     = track.metadata['ARTIST']
        album_artist     = track.metadata['ALBUM_ARTIST']
        # If possible, overwrite the track artist w/ album artist to avoid issues in songs with fts
        track_artist     = album_artist if album_artist else track_artist

        print(f"Current : {track_artist} : {track_name}")

        # Handle errors using a timeout with exponential backoff. Let us define our params first.
        retry_limit = 5     # max number of retries to attempt.
        base_delay  = 1     # initial delay in seconds.
        retries     = 0     # used to track number of retries.

        # Use exponential backoff until request to Spotify API is successful.
        while True:
            try:
                spotify_features = self.spotify_api_extractor.get_spotify_features(track_artist,
                                                                                   track_name,
                                                                                   track_album,
                                                                                   access_token)
                track.update_features(spotify_features)
                return track

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

        # -----------------------------------------------------------------------------------------
        # Step 1 : Metadata Extraction and Spotify API Features Extraction (Parallel)
        # -----------------------------------------------------------------------------------------
        load_dotenv()
        client_id     = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        access_token  = request_access_token(client_id, client_secret)
        filename_list = [os.path.join(self.base_path, filename)
                         for filename in os.listdir(self.base_path)
                         if filename.endswith(('.flac', '.mp3'))]

        result_tracks = run_in_parallel(self._get_metadata_and_spotify,
                             filename_list, access_token,
                             executor_type="thread"
                        )

        self.track_list = [track for track in result_tracks if track is not None]

        # -----------------------------------------------------------------------------------------
        # Step 2 : Essentia Models Extraction
        # -----------------------------------------------------------------------------------------
        start  = time.time()
        result = run_in_parallel(self.audio_feature_extractor.retrieve_all_essentia_features,
                                    self.track_list,
                                    essentia_models_dict,
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
