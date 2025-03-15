"""
In this file I aim to create a simple Python script that uses a .CSV that contains the details for
new tracks that will be added to the dataset with the headers : `track, artist, album` and automates
the process of (1) adding these to a Spotify Playlist (2) Creating a directory with a copy of each
of the tracks.

The .csv file must:
    - Contain all of the mention features as the header. 
    - Be as precise as possible with the naming. One single wrong name could lead to mistakes.

The second (2) point assumes that you:
    - Have downloaded the files for each of the tracks.
    - Have made sure that these files are properly formatted.
    - Have organized your files in a structure such as `Music_Directory/Artist_Name/Album_Name/`
"""
import os
import re
import glob
import time
import random
from typing import Any

import shutil
import unicodedata
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import HTTPError, Timeout
from src.extractors.metadata import MetadataExtractor

from src.utils.parallel import run_in_parallel
from src.utils.sptpy import create_playlist_with_tracks
from src.extractors.spotify_api import SpotifyAPI, request_access_token


class DatasetCreator:
    """..."""

    def __init__(self, csv_path : str, music_dir_path : str, destination_dir : str):
        self.dest_dir       = destination_dir
        self.csv_path       = csv_path
        self.playlist_df    = pd.read_csv(csv_path)
        self.music_dir_path = music_dir_path

        # Reformat the DataFrame so that tracks are kept in a list
        # rather than taking up multiple rows.
        self.playlist_df = self.playlist_df.groupby(["album","artist"],
                                                     as_index=False).agg({"track" : list})
        self.playlist_df.rename(columns={"track" : "track_list"}, inplace=True)

        self.file_dict = {}
        self.file_list = glob.glob(f"{self.music_dir_path}/*/")

        # Create a dictionary in the format of `artist_name : dir_path`. This is done to allow quick
        # access to these paths by only checking for the artist name when comparing with the .csv
        for file in self.file_list:
            artist_name = os.path.basename(os.path.dirname(file))
            artist_name = self.normalize_attribute(artist_name)
            self.file_dict[artist_name] = file


    def normalize_attribute(self, attr_name : str) -> str:
        """Self Explanatory.

        Args:
            attr_name : The string representing the name of the attribute.

        Returns:
            normalized_attr_name: The normalized attribute name with all of the changes made.
        """
        attr_name = unicodedata.normalize("NFKD", attr_name)
        attr_name = "".join([char for char in attr_name if not unicodedata.combining(char)])
        attr_name = re.sub(r'[^a-z0-9]', '', attr_name.lower())
        return attr_name


    def format_tracks(self, track_list: list[str], album_path: str) -> dict[str, str]:
        """Formats tracks into a dictionary with normalized track names as keys and their paths as
        values.  Only `.flac` files are processed.

        Args:
            track_list : The list of tracks received from an album which resides in the MUSIC_DIR.
            album_path : The path to the album from which the tracks were retrieved.

        Returns:
            track_dict : A dictionary mapping normalized track names to their full file paths.
        """
        track_dict = {}

        for track in track_list:
            track_path = os.path.join(album_path, track)

            # Only process files with a .flac extension (case insensitive)
            if track.lower().endswith(".flac"):
                track_name = MetadataExtractor.extract(track_path)["TITLE"]
                track_name = self.normalize_attribute(track_name)

                track_dict[track_name] = track_path

        return track_dict


    def iterate_df(self, copy : bool = True):
        """Iterate through the DataFrame to match tracks within the music directory. Doubles as
        a method to validate that your dataset is matching if `copy = False`. 
        
        Args:
            copy (bool, optional): Determines whether the files will be copied to the destination
                                   folder or not. Defaults to True.
        """
        for row in self.playlist_df.itertuples(index=False):

            artist_name = self.normalize_attribute(row.artist)
            if artist_name not in self.file_dict:
                print(f"Artist '{row.artist}' not found in directory.")
                continue

            row_album   = self.normalize_attribute(row.album)
            artist_path = self.file_dict[artist_name]   # TODO : Try Catch Here for album listdir
            album_list  = os.listdir(artist_path)
            album_match = False

            for album in album_list:
                album_name = self.normalize_attribute(album)

                if row_album in album_name:
                    album_match = True
                    album_path  = os.path.join(artist_path, album)

                    track_list = os.listdir(album_path) # TODO : Try Catch Here for track listdir
                    track_dict = self.format_tracks(track_list, album_path)

                    for track in row.track_list:
                        normalized_track = self.normalize_attribute(track)
                        found_track = False

                        # Forced to iterate this way due to potential problems with track tagging.
                        for track_name, track_path in track_dict.items():
                            if normalized_track in track_name:
                                found_track = True

                                if copy:
                                    os.makedirs(self.dest_dir, exist_ok=True)
                                    shutil.copy(track_path, self.dest_dir)

                        if not found_track:
                            print(f"Track : {track} not found")

            if not album_match:
                print(f"Album '{row.album}' not found for artist '{row.artist}'.")


    def _playlist_helper(self, row : Any, access_token : str):
        # Handle errors using a timeout with exponential backoff. Let us define our params first.
        # TODO Change Printing for logging. Should really do this across most of the code.
        retry_limit = 5     # max number of retries to attempt.
        base_delay  = 1     # initial delay in seconds.
        retries     = 0     # used to track number of retries.

        while True:
            try:
                spotify_features = SpotifyAPI.get_spotify_features( row.artist,
                                                                    row.track,
                                                                    row.album,
                                                                    access_token)

                print(f"Match Found : {row.artist} : {row.track} -> {spotify_features['sp_name']}")
                return spotify_features['uri']

            # If we catch an error, we retry and delay our calls until we find success.
            except (HTTPError, Timeout, ConnectionError) as err:
                print(f"Error retrieving Spotify features for {row.artist} - {row.track}: {err}")

                retries += 1
                if retries > retry_limit:
                    print("Max retries reached. Skipping track.")
                    spotify_features = {}  # or handle as needed
                    break

                # Calculate delay with exponential backoff and some jitter
                delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5)
                time.sleep(delay)


    def add_to_spotify_playlist(self, playlist_name : str, redirect_uri : str):
        """Extracts the URIs for the tracks in the .CSV and creates a playlist with them."""

        # For the sake of efficiency (and running the TPE faster) I will have to reformat
        # the dataframe and acquire it as a list of rows. I commence with that.
        curr_playlist_df = pd.read_csv(self.csv_path)
        row_list         = list(curr_playlist_df.itertuples(index=False))
        print(f"{len(row_list)} tracks to be processed.")
        pprint(row_list)

        # Get the access token so that we can make the call later on.
        load_dotenv()
        client_id     = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        access_token  = request_access_token(client_id, client_secret)

        uri_list = run_in_parallel(self._playlist_helper, row_list,
                                   access_token, executor_type="thread")

        create_playlist_with_tracks(client_id, client_secret, redirect_uri, playlist_name, uri_list)
