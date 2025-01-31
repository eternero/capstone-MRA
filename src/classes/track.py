"""
In this file I aim to develop a 'Track' class which is able to capture all of the features that
have been extracted for the tracks that we'll be using in our dataset.

My main concern with this is wheter we will be standardizing this or not. That is, wheter the class
would look something like this (standardized):

    def __init__(self, timbre, tempo, tonality, key, ... , mood):
        self.timbre = timbre
        self.tempo  = tempo
        ...
        self.mood   = mood

And in the case of being non-standardized, we'd just have a dictionary attribute which saves
all of the features in that dictionary... I guess this wouldn't be bad for now, I'll proceed
with this approach at the moment.
"""
import os
from typing import Any
from essentia.standard import MonoLoader
from src.extractors.metadata import MetadataExtractor
from src.extractors.spotify_api import SpotifyAPI


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

        self.features = {}
        self.metadata = None
        self.metadata_extractor = MetadataExtractor(self.track_path)
        self.spotify_api = SpotifyAPI()

    def set_features(self, features : dict[str, Any]):
        """
        ...
        """
        self.features = features

    def get_features(self):
        """
        ...
        """
        return self.features

    def get_track_path(self):
        """
        ...
        """
        return self.track_path

    def get_track_mono(self):
        """
        ...
        """
        return self.track_mono

    def get_spotify_features(self):
        """
        NOTE : As of now, the only Spotify feature I'm concerned about is the popularity and 
        potentially the release type (album, mixtape, EP, etc...) Not much really.

        NOTE 
        Clean this up a little for the love of God.
        """

        # The metadata is required for this, as we would need track_artist and track_name
        if not self.metadata:
            self.get_track_metadata()

        track_artist = self.metadata['artist']
        track_name   = self.metadata['title']

        spotify_features = self.spotify_api.get_spotify_features(track_artist, track_name)
        self.features.update(spotify_features)
        return spotify_features


    def get_track_metadata(self):
        """
        Retrieves metadata for the track specified by `self.track_path`. Caches the result in 
        `self.metadata`, if `self.metadata` is already set, it is returned without re-processing.

        Returns:
            The metadata dictionary/structure if successfully processed, or None if the file format
            is unsupported or an error occurs.
        """
        # If metadata is already set, return it immediately (caching behavior)
        if not self.metadata:
            try:
                extension = os.path.splitext(self.track_path)[1].lower()

                if extension == ".flac":
                    self.metadata = self.metadata_extractor.process_flac()
                elif extension == ".mp3":
                    self.metadata = self.metadata_extractor.process_mp3()
                else:
                    print(f"Unsupported file format: {extension} in {self.track_path}")

            except Exception as e:
                print(f"Error processing {self.track_path}: {e}")

        return self.metadata


def get_tracks(base_path : str) -> list[MonoLoader]:
    """
    ...
    """
    track_list = []
    audio_dir  = base_path

    for track_path in os.listdir(audio_dir):
        if os.path.isfile(os.path.join(audio_dir, track_path)):
            """            
            NOTE : Still haven't fixed the sample rate hardcoding...
            I should read some of the documentation to find out the drawbacks
            of not providing songs with a 16kHz sample rate as the models mainly require
            i.e. most of the models are trained with songs that have a 16kHz sample rate...
            Does this matter... does this function re-sample in any case...?

            This issue seems to provide some valuable information on it.
                https://github.com/MTG/essentia/issues/1442

            NOTE : Nonetheless, I should go over the parameters for MonoLoader and understand what 
            each of them represent. This could of course allow for more better performance and/or
            even better results

            NOTE : The documentation seems fine with 44kHz (44100), so I guess I'll be testing out
            if there are any differences. `resampleQuality` should stay at zero though.
            """
            track_mono = MonoLoader(filename=f"{base_path}/{track_path}",
                                    sampleRate=44100, resampleQuality=0)()

            curr_track = Track(track_path= track_path, track_mono=track_mono)
            track_list.append(curr_track)

    return track_list
