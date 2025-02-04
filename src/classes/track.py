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
import multiprocessing as mp
from typing import Any, List, Dict
from essentia.standard import MonoLoader
from src.extractors.spotify_api import SpotifyAPI
from src.classes.essentia_models import EssentiaModel
from src.extractors.metadata import MetadataExtractor
from src.extractors.audio_features import FeatureExtractor


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
                track = Track(track_path=track_filename, track_mono=track_mono)
                print(f"Loaded track: {track_filename}")
                return track
            except Exception as e:
                print(f"Error loading track {track_filename}: {e}")
        return None


    def load_tracks(self, num_processes: int = mp.cpu_count()) -> None:
        """
        Loads all audio files from `base_path` using multiprocessing.

        Args:
            num_processes (int, optional): Number of parallel processes to use. 
                                           Defaults to available CPU cores.
        """
        track_filenames = os.listdir(self.base_path)

        # Prepare arguments for multiprocessing
        args = [(filename, self.base_path, self.sample_rate) for filename in track_filenames]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(self._load_single_track, args)

        # Filter out any None results (failed loads)
        self.track_list = [track for track in results if track is not None]


    def run_pipeline(self,
                     essentia_models_dict : Dict[EssentiaModel, List[EssentiaModel]]
                    ) -> List[Track]:
        """
        Processes all tracks in the following order:
          1. Extract metadata
          2. Extract Spotify features (requires metadata)
          3. Run any additional steps

        Args:
            essentia_models_dict : This is a dictionary that contains pairs of 
                {Essentia Embeddings : List[Essentia Model]}... This is because multiple models can
                depend on the same embeddings, and containing them as such provides an efficient 
                approach..
        """
        if not self.track_list:
            self.load_tracks()

        # -----------------------------------------------------------------------------------------
        # Step 1 : Metadata Extraction and Spotify API Features Extraction
        # -----------------------------------------------------------------------------------------
        for track in self.track_list:
            metadata = self.metadata_extractor.extract(track.track_path)
            track.update_metadata(metadata)

            # Once we've got the metadata, we can proceed to get the Spotify API Features
            track_name   = track.metadata['TITLE']
            track_artist = track.metadata['ARTIST']
            spotify_features = self.spotify_api_extractor.get_spotify_features(track_artist,
                                                                               track_name)
            track.update_features(spotify_features)


        # -----------------------------------------------------------------------------------------
        # Step 2 : Essentia Models Extraction
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
