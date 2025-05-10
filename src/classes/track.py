"""
...
"""
import os
import time
import copy
import random
import itertools
from typing import Any, List, Dict, Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import HTTPError, Timeout


import essentia
from src.extractors.metadata import MetadataExtractor
from src.extractors.audio_features import FeatureExtractor
from src.extractors.spotify_api import SpotifyAPI, request_access_token
from src.utils import run_in_parallel, pool_segments
from src.classes.essentia_containers import FeatureTask


# DISABLE LOGGING. ANNOYING!
essentia.log.infoActive = False
essentia.log.warningActive = False


@dataclass
class Track:
    """
    The Track object should encapsulate all of the necessary elements that we've determined to be
    pertinent to describing and analyzing a song. This includes metadata, features extracted using
    Essentia, and other information acquired from the SpotifyAPI.

    Attributes:
        track_path : The path which points to the location of the song file.
        track_mono : The MonoLoader (or numpy array) which represents the song.

        features   : The dictionary which stores the features that have been acquired for the track
        metadata   : The metadata of the track. Includes track name, album name, artist name etc...
    """
    track_path    : str
    track_mono_16 : np.ndarray     = field(default_factory=list)
    track_mono_44 : np.ndarray     = field(default_factory=list)

    features      : Dict[str, Any] = field(default_factory=dict)
    metadata      : Dict[str, Any] = field(default_factory=dict)
    segment_start : int            = field(default_factory=int)


    def segment_track(self,
                      seg_num           : int,
                      seg_start         : int,
                      new_track_mono_16 : np.ndarray = None,
                      new_track_mono_44 : np.ndarray = None) -> "Track":
        """Creates a segmented copy of a track."""

        # Set the segment_num and segment_start
        new_metadata                  = self.metadata.copy()
        new_metadata['segment_num']   = seg_num
        new_metadata['segment_start'] = seg_start
        inherited_features            = copy.deepcopy(self.features)

        track_segment = replace(self,
                       metadata       = new_metadata,
                       segment_start  = seg_start,
                       features       = inherited_features,
                       track_mono_16  = new_track_mono_16,
                       track_mono_44  = new_track_mono_44,)

        return track_segment


    def create_segments(self) -> list["Track"]:
        """Returns segmented track objects"""
        if 'length' not in self.metadata:
            raise KeyError('`length` attribute must be present in track metadata')

        track_length   = self.metadata['length']
        seg_track_list = []
        segment_size   = 10

        for sec in range(0, track_length, 60):

            # Skip frames that might be too small. This should only occur for final-minute frames.
            if track_length - sec < segment_size:
                continue

            # Acquire the random number, and then acquire the segmented frame for each sample rate.
            lower_bound = sec
            upper_bound = min(sec + 50,track_length - 10) # Make sure to stay in bounds in the track
            start_sec   = random.randint(lower_bound, upper_bound)

            seg_num     = sec // 60
            seg_track   = self.segment_track(seg_num   = seg_num,
                                             seg_start = start_sec)

            seg_track_list.append(seg_track)

        return seg_track_list


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


    def _get_metadata_and_spotify(self, track_path : Track, access_token : str) -> list[Track]:
        """Acquires the metadata and Spotify API features for a single track.

        Args:
            track_path   : The path to the track for which we will be acquiring its features.
            access_token : The access token needed to make the Spotify API requests.
        """
        # Initialize a `Track` object.
        track    = Track(track_path=track_path)
        metadata = MetadataExtractor.extract(track.track_path)
        track.metadata.update(metadata)

        # Once we've got the metadata, we can proceed to get the Spotify API Features
        track_name       = track.metadata['title']
        track_album      = track.metadata['album']
        track_artist     = track.metadata['artist']


        # Handle errors using a timeout with exponential backoff. Let us define our params first.
        retry_limit = 5     # max number of retries to attempt.
        base_delay  = 1     # initial delay in seconds.
        retries     = 0     # used to track number of retries.


        # Use exponential backoff until request to Spotify API is successful.
        while True:
            try:
                spotify_features = SpotifyAPI.get_spotify_features(track_artist,track_name,
                                                                   track_album,access_token)
                track.metadata.update(spotify_features)
                break

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


        # Once we've got all of our Metadata and Spotify API Attributes, we can segment our track.
        return track.create_segments()

    def run_pipeline(self, essentia_task_list : List[FeatureTask],
                     additional_tasks         : list[Callable] = None,
                     only_track               : bool = False,
                     pooling                  : bool = False) -> List[Track]:
        """
        Processes all tracks in the following order:
          1. Extract metadata and spotify api features
          2. Extracts Essentia features, whether it be from Models or Algorithms.
          3. Handles any additional tasks to finsih off.

        The first two steps are done concurrently, while the 'additioinal tasks' are simply
        any tasks which could not be executed concurrently. For now, this only applies to HarmoF0.

        Args:
            essentia_task_list : A list of `FeatureTask` objects. These can be either Essentia
                                 Models or Algorithms.
            additional_tracks  : A list of additional tasks which cannot be run concurrently, these
                                 must all take `track_mono` as their input and return a dictionary
                                 of the features which were computed.
            only_track         : Determines whether only the Track Metadata will be extracted.
                                 Defaults to False. If True, no other tasks will run.
        """

        # -----------------------------------------------------------------------------------------
        # Step 1 : Metadata Extraction and Spotify API Features Extraction (Parallel)
        # -----------------------------------------------------------------------------------------
        load_dotenv()
        client_id     = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        access_token  = request_access_token(client_id, client_secret)


        # Check if the base path is a directory or not, to take the appropriate action.
        if os.path.isdir(self.base_path):
            filename_list = [os.path.join(self.base_path, filename)
                            for filename in os.listdir(self.base_path)
                            if filename.endswith(('.flac', '.mp3'))]

        else:
            filename_list = [self.base_path] if self.base_path.endswith(('.flac', '.mp3')) else []

        # If there are no tracks to be processed, then we must raise an error.
        if not filename_list:
            raise ValueError("Base Path must contain (or be) at least one .mp3 or .flac file")

        result_tracks = run_in_parallel(self._get_metadata_and_spotify,
                             filename_list, access_token,
                             executor_type="thread"
                        )

        flat_track_list = list(itertools.chain(*result_tracks))
        self.track_list = [track for track in flat_track_list if track is not None]

        # -----------------------------------------------------------------------------------------
        # Step 2 : Essentia Models Extraction
        # -----------------------------------------------------------------------------------------
        if not only_track:

            if len(self.track_list) <= 5:        # If we only have a few segments... we're better
                for track in self.track_list:    # off doing this than using multiprocessing.
                    track = FeatureExtractor.retrieve_all_essentia_features(track,
                                                                            essentia_task_list)
            else:
                result_tracks = run_in_parallel(FeatureExtractor.retrieve_all_essentia_features,
                                            self.track_list,
                                            essentia_task_list,
                                            num_workers   = 10,
                                            executor_type = "process",
                            )

                self.track_list = [track for track in result_tracks if track is not None]

        # -----------------------------------------------------------------------------------------
        # Step 3 : Handle any tasks that can't be parallelized. Must take `track_mono` as input.
        # -----------------------------------------------------------------------------------------
        if additional_tasks is None or only_track:
            additional_tasks = []

        for task in additional_tasks:
            for track in self.track_list:
                task_features = task(track.track_mono_16)
                track.features.update(task_features)


        # -----------------------------------------------------------------------------------------
        # Step 4 : Pool together the track segments
        # -----------------------------------------------------------------------------------------
        if pooling:
            self.track_list = pool_segments(self.track_list)

        return self.track_list


    def get_track_dataframe(self) -> pd.DataFrame:
        """
        This method is to be used only once the `run_pipeline()` method has been run and
        succesfully completed. It will compile all of the track features into one huge dataframe.
        These features will only be acquired from the dictionary stored in `track.features`.

        Additionally, the tracks will be 'tagged' with the first four columns, which will include

                    FILENAME  ,   ARTIST  ,   TITLE  ,   ALBUM

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
        for track in self.track_list:
            row = {}
            row.update(track.metadata)
            row.update(track.features)
            rows.append(row)

        # Create DataFrame from list of row dictionaries.
        df = pd.DataFrame(rows)
        return df


