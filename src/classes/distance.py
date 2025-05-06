"""Everything that has to do with distance."""
import ast
from copy import deepcopy
from typing import Callable
from dataclasses import dataclass, asdict
from src.utils.parallel import run_in_parallel

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------------------------------
#  Define as Global the final selected features...
# -------------------------------------------------------------------------------------------------
num_features          = {'electronic_effnet' : 5.0, 'instrumental_effnet' : 5.0,
                         'acoustic_effnet'   : 3.0, 'female_effnet'    : 2.0, 'bpm' : 1.0,

                         'happy_effnet'      : 3.0, 'danceable_effnet' : 3.0,
                         'aggressive_effnet' : 3.0, 'relaxed_effnet'   : 3.0,
                         'party_effnet'      : 3.0, 'sad_effnet'       : 3.0,
                        }
dim_features          = {'mfcc_mean'  : 0.01, 'discogs_effnet_embeddings' : 5.0}
z_score_normalization = StandardScaler().fit_transform


@dataclass
class DistTrack:
    """
    A simplified Track object used to store the Distance of a Track relative to the Input Track.
    Essentially a useful helper to keep the `DistPipeline` results nice and clean.

    Attributes:
        - track_artist   : The artist who performs the track.
        - track_album    : The album (or EP, Compilation, Single, etc.) to which a track belongs.
        - track_title    : The title of the track.
        - spotify_uri    : The Spotify URI of the Track, can be used to embed as a Spotify Link.
        - track_distance : The distance of the track relative to the input track.
    """
    track_artist   : str
    track_album    : str
    track_title    : str
    spotify_uri    : str
    track_distance : float


class DistMethods:
    """A collection of distance methods for both numerical and dimensional features."""

    # ---------------------------------------------------------------------------------------------
    #  Define the numerical distance methods first
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _minkowski_numerical(input_track        : pd.Series,
                             comp_track         : pd.Series,
                             numerical_features : dict[str, float],
                             p                  : float):
        """
        Implementation for the Minkowski Distance Equation. This method iterates through each
        feature in the `numerical_features` dictionary and gathers the cumulative distance between
        the two tracks - `input_track` and `comp_track` - for all features.

        Args:
            input_track        : The track provided by the user for which we will be searching for
                                 other similar tracks.
            comp_track         : The current track from our dataset which we will be comparing the
                                 input to.
            numerical_features : The dictionary of features that we will be using to compare the two
                                 tracks alongside their weights.
            p                  : The float value representing the power of the distance.

        Returns:
            distance : The cumulative distance of the features between the two tracks.
        """

        difference = 0
        for feature, weight in numerical_features.items():
            input_feat = input_track[feature]
            comp_feat  = comp_track[feature]

            difference += ((input_feat - comp_feat)**p) * weight

        # Get the actual distance, rather than the weighted squared differences...
        distance = difference**(1/p)
        return float(distance)


    @staticmethod
    def euclidean_numerical(input_track        : pd.Series,
                            comp_track         : pd.Series,
                            numerical_features : dict[str, float]):
        """
        Implementation for Euclidean Distance. Refer to `_minkowski_numerical()` for more details.
        """
        return DistMethods._minkowski_numerical(input_track        = input_track,
                                                comp_track         = comp_track,
                                                numerical_features = numerical_features,
                                                p                  = 2)


    @staticmethod
    def manhattan_numerical(input_track        : pd.Series,
                            comp_track         : pd.Series,
                            numerical_features : dict[str, float]):
        """
        Implementation for Manhattan Distance. Refer to `_minkowski_numerical()` for more details.
        """
        return DistMethods._minkowski_numerical(input_track        = input_track,
                                                comp_track         = comp_track,
                                                numerical_features = numerical_features,
                                                p                  = 1)

    @staticmethod
    def cosine_numerical(input_track        : pd.Series,
                         comp_track         : pd.Series,
                         numerical_features : dict[str, float]):
        """
        Acquires the **Cosine Similarity** for a given set of features between two tracks,
        the `input_track` which is provided by the user and the `comp_track` from our dataset.
        Not to be confused with the **Cosine Distance**, they're not the same!

        Args:
            input_track        : The track provided by the user for which we will be searching for
                                 other similar tracks.
            comp_track         : The current track from our dataset which we will be comparing the
                                 input to.
            numerical_features : The dictionary of features that we will be using to compare the two
                                 tracks alongside their weights.

        Returns:
            cosine_similarity  : The cosine similarity for the feature vector between the two tracks.
        """

        # Get our stuff
        feature_keys   = list(numerical_features.keys())
        input_features = [input_track.get(feat, 0.0) for feat in feature_keys]
        comp_features  = [comp_track.get(feat, 0.0)  for feat in feature_keys]

        #  Calculate Similarity
        input_norm     = np.linalg.norm(input_features)
        comp_norm      = np.linalg.norm(comp_features)
        norm_product   = input_norm * comp_norm

        dot_product    = np.dot(input_features, comp_features)
        similarity     = np.clip(dot_product / norm_product, -1.0, 1.0)

        cosine_dist    = 1.0 - similarity
        return float(cosine_dist)

    # ---------------------------------------------------------------------------------------------
    #  Define the dimensional distance methods now
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def euclidean_dimensional(input_track          : pd.Series,
                              comp_track           : pd.Series,
                              dimensional_features : dict[str, float]):
        """The same as `euclidean_numerical()` but for dimensional features."""

        dist_score = 0
        for feature, weight in dimensional_features.items():

            # Have to use `ast.literal_eval()` since the dimensional features are saved as strings.
            input_feat = input_track[feature]
            comp_feat  = comp_track[feature]

            dist_score += (np.linalg.norm(input_feat - comp_feat)) * weight

        return float(dist_score)


    @staticmethod
    def cosine_dimensional(input_track          : pd.Series,
                           comp_track           : pd.Series,
                           dimensional_features : dict[str, float]):
        """The same as `cosine_numerical()` but for dimensional features."""

        dist_score = 0
        for feature, weight in dimensional_features.items():

            # Unpack features
            input_feature = input_track.get(feature)
            comp_feature  = comp_track.get(feature)

            input_norm    = np.linalg.norm(input_feature)
            comp_norm     = np.linalg.norm(comp_feature)

            cosine_norm   = input_norm * comp_norm
            cosine_dot    = np.dot(input_feature, comp_feature)
            similarity    = np.clip(cosine_dot / cosine_norm, -1.0, 1.0)

            # Convert similarity to distance
            feat_distance = 1.0 - similarity
            dist_score    += abs(feat_distance * weight)

        return float(dist_score)

    @staticmethod
    def pool_dataframe(track_df   : pd.DataFrame,
                       num_feats  : list[str],
                       dim_feats  : list[str],
                      ):
        """
        Aggregates the features of the Track Dataframe by their mean values. Makes a distinction
        between numerical features (`float`), dimensional features (`list[float]`) and other
        features that could be metadata or that could simply not be pooled by mean.

        Args:
            track_df  : All of these
            num_feats : are self
            dim_feats : explanatory

        Returns:
            pooled_df : The aggregated dataframe by the mean of the features.
        """

        def mean_array(series : pd.Series):
            """Helper function to get the mean of vector features for tracks in our dataframe."""
            return np.stack(series.values, axis=0).mean(axis=0)


        track_df_cols = track_df.columns.values
        metadata_cols = [feat for feat in track_df_cols
                        if feat not in num_feats and
                           feat not in dim_feats
                        ]

        # Build an aggregate dict with a function as the value to handle different types of feature.
        agg_map = {}
        agg_map.update({c: 'first' for c in metadata_cols})     # metadata  → stays the same across all
        agg_map.update({c: 'mean' for c in num_feats})          # numerical → mean
        agg_map.update({c: mean_array for c in dim_feats})      # vector    → np.mean() w/ helper

        # Perform the pooling / grouping.
        pooled = track_df.groupby('filename', as_index=True).agg(agg_map)
        return pooled


class DistPipeline:
    """
    Runs the entire procedure of acquiring the distance between an input track and a track dataset.
    Handles normalization of dataset and

    Args:
        input_filename        : The string with the filename of the chosen input track.
        track_dataset_path    : The path (str) to our .csv track dataset.

        numerical_dist        : The method to be used for numerical (`float`) features.
        dimensional_dist      : The method to be used for dimensional (`list[float]`) features.

        numerical_features    : The dictionary containing the numerical features to be used and
                                their weights. Defaults to None.
        dimensional_features  : The dictionary containing the dimensional features to be used and
                                their weights. Defaults to None.
        expand_dimensional    : The boolean which determines whether the dimensional features will
                                be expanded into numerical features. Defaults to false.

        normalize_numerical   : The method to normalize the numerical features. Defaults to None.
        normalize_dimensional : The method to normalize the dimensional features. Defaults to None.

        pooling               : The boolean determining whether the segmented dataset will be
                                pooled by the mean values of its features or not. Defaults to False.

        save_df               : Provides the option to save the modified dataframe. This is some
                                micro-spaghetti-retrospective-code. Used to save pooled dataframe.
    """

    def __init__(self,
                 input_track_df        : pd.DataFrame,
                 track_dataset_path    : str,
                 numerical_dist        : Callable,
                 dimensional_dist      : Callable,
                 numerical_features    : dict[str, float] = None,
                 dimensional_features  : dict[str, float] = None,
                 expand_dimensional    : bool             = False,
                 normalize_numerical   : Callable         = None,
                 normalize_dimensional : Callable         = None,
                 pooling               : bool             = False,
                 save_df               : bool             = False
                ):

        self.input_track_df        = input_track_df
        self.track_dataset_path    = track_dataset_path
        self.numerical_features    = numerical_features
        self.dimensional_features  = dimensional_features
        self.numerical_dist        = numerical_dist
        self.dimensional_dist      = dimensional_dist
        self.normalize_numerical   = normalize_numerical
        self.normalize_dimensional = normalize_dimensional
        self.pooling               = pooling

        # Load the dataset into a DataFrame. Copy it into another dataframe that is to be modified.
        self.original_track_df     = pd.read_csv(track_dataset_path)
        self.mod_track_df          = deepcopy(self.original_track_df)
        self.mod_track_df          = self.mod_track_df.set_index('filename')

        # -----------------------------------------------------------------------------------------
        #  Handle all the normalization stuff below.
        # -----------------------------------------------------------------------------------------
        if expand_dimensional is True and normalize_dimensional:
            raise ValueError('Cannot use `normalize_dimensional`'\
                             'on dimensional features which have been expanded.')

        # Handle expansion of dimensional features
        if expand_dimensional:
            self.mod_track_df, expanded_fts = DistPipeline.expand_dimensional_feature(
                                                track_df     = self.mod_track_df,
                                                feature_list = self.dimensional_features)

            # Clear all dimensional features, as these have been expanded.
            self.dimensional_features = None
            self.numerical_features.update(expanded_fts)

        # Take the features as lists so we can normalize and do whatever on them.
        num_ft_list = (list(self.numerical_features.keys())
                       if self.numerical_features else [])
        dim_ft_list = (list(self.dimensional_features.keys())
                       if self.dimensional_features else [])

        if self.normalize_numerical and num_ft_list is True:
            self.mod_track_df[num_ft_list] = normalize_numerical(self.mod_track_df[num_ft_list])

        if self.normalize_dimensional and dim_ft_list is True:
            self.mod_track_df[dim_ft_list] = normalize_numerical(self.mod_track_df[dim_ft_list])

        # Parse the dimensional features if there are any.
        if dim_ft_list:
            for dim_ft in dim_ft_list:
                self.mod_track_df[dim_ft] = (self.mod_track_df[dim_ft]
                                                .apply(ast.literal_eval)
                                                .apply(np.array)
                                                )

        if pooling is True:
            # If everything is good, proceed.
            self.mod_track_df   = DistMethods.pool_dataframe(track_df  = self.mod_track_df,
                                                             num_feats = num_ft_list,
                                                             dim_feats = dim_ft_list)

            self.input_track_df = DistMethods.pool_dataframe(track_df  = self.input_track_df,
                                                             num_feats = num_ft_list,
                                                             dim_feats = dim_ft_list)

        if save_df is True:
            # Unparse the dim features to save the df... Terrible code, god please kill me.
            copy_df = deepcopy(self.mod_track_df)
            for dim_ft in dim_ft_list:
                copy_df[dim_ft] = copy_df[dim_ft].apply(np.ndarray.tolist)
            copy_df.to_csv('pooled_dataset.csv', index=True)


    @staticmethod
    def expand_dimensional_feature(track_df : pd.DataFrame, feature_list : dict[str, float]):
        """
        Expands dimensional features, effectively turning them into new numerical features.\n
        Usage of this is discouraged, given that it will bring on the Curse of Dimensionality.

        Args:
            track_df     : The DataFrame containing all of the tracks which will be used for recs.
            feature_list : The dictionary of features to be expanded.

        Returns:
            expanded_df  : The DataFrame containing the expanded features.
            expanded_fts : The dictionary containing the newly expanded features.
        """
        for feature in feature_list:
            expanded         = track_df[feature].apply(ast.literal_eval).apply(pd.Series)
            expanded.columns = [f"{feature}_{i}" for i in range(expanded.shape[1])]

            # TODO : Have to actually assign the weight of the original features to these
            expanded_fts     = {feature : 1.0 for feature in expanded.columns}
            expanded_df      = pd.concat([track_df, expanded], axis=1)

        return expanded_df, expanded_fts


    def run_pipeline(self, top_n: int = 20) -> list[dict]:
        """Compute the top_n nearest tracks to the input track."""
        result = []

        for _, comp_track in self.mod_track_df.iterrows():

            best_dist = float('inf')
            for _, input_track in self.input_track_df.iterrows():
                distance = 0.0
                if self.numerical_features:
                    distance += self.numerical_dist(input_track, comp_track, self.numerical_features)
                if self.dimensional_features:
                    distance += self.dimensional_dist(input_track, comp_track, self.dimensional_features)

                if distance < best_dist:
                    best_dist = distance

            result.append(
                DistTrack(
                    track_artist   = comp_track.artist,
                    track_album    = comp_track.album,
                    track_title    = comp_track.title,
                    spotify_uri    = comp_track.uri,
                    track_distance = best_dist,
                )
            )


        # NOTE : If we're given a track that is already in the dataset, then it will be it's own top
        # result. Not sure how to deal with that yet! Metadata checking could work.
        result.sort(key=lambda dt: dt.track_distance)
        return [asdict(track) for track in result][:top_n]


    def get_dist_helper(self, comp_track : pd.Series) -> DistTrack:
        """
        Helper method to acquire Distance Values for Tracks in Parallel.

        NOTE : This method assumes that pooling will be used - since we will indeed be using it for
        our final demo, however it won't work well if pooling isn't used given that there will be
        duplicate `DistTrack` objects for the same track w/ different segments.

        Some post-processing could be done in that case, but that is to say that this method
        works at its best if pooling is used.
        """

        best_dist = float('inf')
        for _, input_track in self.input_track_df.iterrows():

            distance = 0.0
            if self.numerical_features:
                distance += self.numerical_dist(input_track, comp_track,
                                                self.numerical_features)
            if self.dimensional_features:
                distance += self.dimensional_dist(input_track, comp_track,
                                                  self.dimensional_features)

            if distance < best_dist:
                best_dist = distance

        result_track = DistTrack(
                    track_artist   = comp_track.artist,
                    track_album    = comp_track.album,
                    track_title    = comp_track.title,
                    spotify_uri    = comp_track.uri,
                    track_distance = best_dist,
                )
        return result_track


    def run_pipeline_parallel(self, top_n: int = 20) -> list[dict]:
        """Copmute the top_n nearest tracks for the input track with Multi Threading."""

        comp_track_list = list([comp_track for _,comp_track in self.mod_track_df.iterrows()])
        result_list     = run_in_parallel(func          = self.get_dist_helper,
                                          item_list     = comp_track_list,
                                          executor_type = "thread")


        result_list.sort(key=lambda dt: dt.track_distance)
        return [asdict(track) for track in result_list][:top_n]
