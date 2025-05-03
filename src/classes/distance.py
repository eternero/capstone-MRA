"""Everything that has to do with distance."""
import ast
import logging
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define columns to be used for Pooling... Essentially all `float` or `list[float]` features.
NUM_COLS  = ['approachability_effnet', 'approachable_effnet', 'engagement_effnet', 'engaging_effnet',
            'danceable_effnet', 'aggressive_effnet', 'happy_effnet', 'party_effnet', 'relaxed_effnet',
            'sad_effnet', 'acoustic_effnet', 'electronic_effnet', 'instrumental_effnet', 'female_effnet',
            'mfcc_avg_energy', 'mfcc_peak_energy', 'avg_dissonance', 'avg_rollof', 'bpm', 'avg_energy',
            'tonal_effnet',	'bright_effnet', 'bright_nsynth_effnet', 'pitch_var', 'pitch_salience'
            ]
DIM_COLS  = ['tristimulus','mfcc_mean', 'mfcc_std', 'band_mean', 'band_std','pitch_hist']


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
                           feat not in dim_feats and
                           feat != 'filename'
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
    """

    def __init__(self,
                 input_filename        : str,
                 track_dataset_path    : str,
                 numerical_dist        : Callable,
                 dimensional_dist      : Callable,
                 numerical_features    : dict[str, float] = None,
                 dimensional_features  : dict[str, float] = None,
                 expand_dimensional    : bool             = False,
                 normalize_numerical   : Callable         = None,
                 normalize_dimensional : Callable         = None,
                 pooling               : bool             = False,
                ):

        self.input_filename        = input_filename
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
            self.mod_track_df = DistMethods.pool_dataframe(track_df  = self.mod_track_df,
                                                           num_feats = num_ft_list,
                                                           dim_feats = dim_ft_list)

        # For flexibility, the Input Track will always be a DataFrame.
        input_temp = self.mod_track_df.loc[self.input_filename]
        if isinstance(input_temp, pd.Series):       # Here we handle the case in which it was turned
            input_temp = input_temp.to_frame().T    # into a pd.Series due to the pooling.

        # Do this for explicit type hinting.
        self.input_df : pd.DataFrame = input_temp


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


    def run_pipeline(self, top_n : int = 21):
        """
        Acquires the `top_n` recommendations for our `input_df` (input track) based on the
        attributes passed in to the `DistPipeline` object.

        By iterating through each track in our dataframe and comparing it to the input track with
        the given Distance Metrics and/or Normalization Methods we gather the resulting distance
        for each track in the dataset and then sort by the closest ones to the input track.

        The input track is handled as a dataframe given that the `self.track_df` is segmented
        by default, meaning that a single track would be represented by multiple rows of segments,
        thus a DataFrame.

        If `DistPipeline` object is decided to be pooled by passing in `pooling = True` then those
        segments become a single row (`pd.Series`) which we then covert to a `pd.DataFrame` of a
        single row to keep the workflow simple.
        """

        # First, make sure that our feature dictionaries are not None.
        self.numerical_features   = {} if not self.numerical_features \
                                       else self.numerical_features

        self.dimensional_features = {} if not self.dimensional_features \
                                       else self.dimensional_features

        result_dict               = {} # Save our results here of course.

        # Iterate through all the tracks in the DataFrame to compare them.
        for _, comp_track in self.mod_track_df.iterrows():
            track_name = "-".join([comp_track['artist'], comp_track['title']])

            # Iterate through each segment in the Input DataFrame
            for _, input_segment in self.input_df.iterrows():

                total_distance = 0
                if self.numerical_features:
                    total_distance += self.numerical_dist(input_segment, comp_track,
                                                        self.numerical_features)

                if self.dimensional_features:
                    total_distance += self.dimensional_dist(input_segment, comp_track,
                                                        self.dimensional_features)

                result_dict[track_name] = min(total_distance, result_dict.get(track_name, float("inf")))


        # Log class attributes...
        excluded_attributes = {"original_track_df", "mod_track_df", "input_df"}
        logging.info(
            "Attributes: %s",
            ', '.join(f"{k}={v}" for k, v in vars(self).items()
                      if k not in excluded_attributes)
        )

        # Log top similar tracks.
        output = dict(sorted(result_dict.items(), key=lambda item: item[1]))
        for ix, items in enumerate(output.items()):
            if ix == 0:
                continue

            if ix >= top_n:
                break
            key, val = items
            logging.info("%d. %s - %f", ix, key, val)
        logging.info("\n"*2)

        # Currently we're printing, but the top tracks can be returned, or this can be
        # altered so that


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------------
    # Define tracks to be used for testing.
    # ---------------------------------------------------------------------------------------------
    testing_tracks = [
        '03 Have You Got It In You-.flac',
        '01 - AAA Powerline.flac',
        '02. Frou Frou - Breathe In.flac',
        '02 Archangel.flac',
        '08 - Sad World.flac',
        '03 Alive in the Septic Tank.flac',
        '15._maddington.flac',
        '13 - Boldy James - Pots and Pans.flac',
        '02 - Unfeeling.flac'
    ]

    # ---------------------------------------------------------------------------------------------
    # Normalization Methods
    # ---------------------------------------------------------------------------------------------
    z_score_normalization = StandardScaler().fit_transform

    # ---------------------------------------------------------------------------------------------
    # HarmoF0 Pitch Features                                                                       |
    # ---------------------------------------------------------------------------------------------
    #   - Pitch Var is absolutely terrible. Not to be used at all.
    #   - Pitch Salience (Not HarmoF0, Essentia)... most recommendations were bad.
    #   - Pitch Mean and Pitch Hist were decent. Got to find the soft spot for their weights.
    # ---------------------------------------------------------------------------------------------
    harmof0_num_feats = {'pitch_var' : 1.0, 'pitch_mean' : 1.0}
    harmof0_dim_feats = {'pitch_hist' : 1.0}

    # ---------------------------------------------------------------------------------------------
    # Effnet High Level / Subjective Features                                                      |
    # ---------------------------------------------------------------------------------------------
    #   Not very great recommendations. These either need some tuning, or to not be used at all.
    #   The only somewhat good thing, is that at times they're able to recognize an artists style
    #   and recommend that very same artist... but the Handpicked Set No.2 can also do this without
    #   the plethora of inconsistencies.
    #
    #   The only other thing to mention is that they yield very low values, so I guess their impact
    #   wouldn't be much... but then again, why add something that sucks?
    #
    #   Performs a bit better when the redundant features are removed, so that is to be considered.
    # ---------------------------------------------------------------------------------------------
    effnet_high_level_feats   = {'happy_effnet' : 1.0, 'danceable_effnet': 1.0,
                                 'aggressive_effnet': 1.0, 'relaxed_effnet': 1.0,
                                 'party_effnet': 1.0, 'sad_effnet': 1.0,
                                }

    # ---------------------------------------------------------------------------------------------
    # Effnet Tonal Features                                                                        |
    # ---------------------------------------------------------------------------------------------
    #   Not sure what to say about these. Nothing to really observe. The values for the numerical
    #   features are rather low – so I'd say that when used all together these are dominated by the
    #   tristimulus.
    #
    #   Either way... the results are not good. Not horrendous, but simply not good. Maybe these
    #   could be used with relatively low weights, since sometimes the Top 1-3 results are decent,
    #   but overall these are not as good as others.
    #
    #   - Addind the `avg_dissonance` makes it a bit better surprisingly. Still not good, but
    #     somewhat more considerable.
    #   - The `tonal` effnet value could probably just be removed... since its mean
    #     is literally like 94... so it just marks everything as tonal ffsk.
    #   - Have to do more thorough testing of `bright_effnet` vs `bright_nsynth`
    # ---------------------------------------------------------------------------------------------
    tonal_num_features        = {'tonal_effnet' : 1.0, 'bright_effnet' : 1.0, 'avg_dissonance' : 1.0}
    tonal_dim_features        = {'tristimulus' : 1.0}

    # ---------------------------------------------------------------------------------------------
    # Effnet Social Features                                                                       |
    # ---------------------------------------------------------------------------------------------
    #   Honestly pretty good for just two numerical features. Excluding the `popularity` feature
    #   from the Spotify API is better in contrast to using it...
    # ---------------------------------------------------------------------------------------------
    social_effnet_features    = {'approachability_effnet':1.0, 'engagement_effnet':1.0}

    # ---------------------------------------------------------------------------------------------
    # Handpicked Features Set #1 - Starting with just a few.                                       |
    # ---------------------------------------------------------------------------------------------
    handpicked_num_features_1 = { 'electronic_effnet' : 1.0,
                                 'instrumental_effnet' : 1.0, 'female_effnet' : 1.0,  'bpm' : 1.0
                                }
    handpicked_dim_features_1 = {'mfcc_mean' : 0.10, 'mfcc_std' : 0.05, 'tristimulus' : 1.0}    # MFCC Features have lower weight
                                                                                                # due to their high dimensionality
                                                                                                # more on this on my .md notes...
    # ---------------------------------------------------------------------------------------------
    # Handpicked Features Set #2 - A few extra features based on previous testing.                 |
    # ---------------------------------------------------------------------------------------------
    #       - Way better results. Can't imagine what this would be able to do with either tuning
    #         the parameters or using contrastive learning.
    #       - Still ways to go, and this does not yet implement nearly half of the extracted
    #         features (not necessary to use them either, but must be tested).
    #       - It does not implement the Pitch Histogram or tristimulus either. Both could be
    #         implemented and would surely be of benefit.
    # ---------------------------------------------------------------------------------------------
    handpicked_num_features_2 = { 'electronic_effnet' : 1.0, 'instrumental_effnet' : 1.0, 'acoustic_effnet' : 0.8,
                                  'female_effnet' : 1.0, 'bpm' : 0.65 }
    handpicked_dim_features_2 = {'mfcc_mean' : 0.06, 'mfcc_std' : 0.15}     # Not adding tristimulus yet.

    # ---------------------------------------------------------------------------------------------
    # Handpicked Features Set #3 - Now with Discogs Effnet Embeddings as well + Weight Changes     |
    # ---------------------------------------------------------------------------------------------
    # The addition of the Discogs Effnet Embeddings are damn near miraculous, but they only work
    # well with pooling. Pooling has made it so that other metrics like MFCC Mean and Std absolutely
    # suck... but after all, they weren't damn near as good as the embeddings.
    #
    # These didn't add up all too well anyways, the numerical features need to be tweaked, so I'll
    # just move on to the next round of testing.
    # ---------------------------------------------------------------------------------------------
    handpicked_num_features_3 = { 'electronic_effnet' : 1.2, 'instrumental_effnet' : 1.2, 'acoustic_effnet' : 0.8,
                                  'female_effnet' : 0.5, 'bpm'     : 0.40}
    handpicked_dim_features_3 = {'mfcc_mean'  : 0.04}

    # ---------------------------------------------------------------------------------------------
    # Handpicked Features Set #4 - MFCC Features Removed or Decreased Weights. Num Features upped  |
    # ---------------------------------------------------------------------------------------------
    # Satisfactory results, but some tweaks can be made. Not too sure how high the weight for the
    # embeddings should be – sure, they do provide the best results, however smaller features also
    # help level things out. Using optimization or triplet loss should help out figuring the values.
    # ---------------------------------------------------------------------------------------------
    handpicked_num_features_4 = {'electronic_effnet' : 5.0, 'instrumental_effnet' : 5.0, 'acoustic_effnet' : 3.0,
                                 'female_effnet'     : 2.0, 'bpm'     : 1.0,

                                 'happy_effnet'      : 3.0, 'danceable_effnet' : 3.0,
                                 'aggressive_effnet' : 3.0, 'relaxed_effnet'   : 3.0,
                                 'party_effnet'      : 3.0, 'sad_effnet'       : 3.0,

                                }
    handpicked_dim_features_4 = {'mfcc_mean'  : 0.05}

    # ---------------------------------------------------------------------------------------------
    # Handpicked Features Set #4 - MFCC Features Removed or Decreased Weights. Num Features upped  |
    # ---------------------------------------------------------------------------------------------
    # Satisfactory results, but some tweaks can be made. Not too sure how high the weight for the
    # embeddings should be – sure, they do provide the best results, however smaller features also
    # help level things out. Using optimization or triplet loss should help out figuring the values.
    # ---------------------------------------------------------------------------------------------
    handpicked_num_features_5 = {'electronic_effnet' : 5.0, 'instrumental_effnet' : 5.0,
                                 'acoustic_effnet'   : 3.0, 'female_effnet'    : 2.0, 'bpm' : 1.0,

                                 'happy_effnet'      : 3.0, 'danceable_effnet' : 3.0,
                                 'aggressive_effnet' : 3.0, 'relaxed_effnet'   : 3.0,
                                 'party_effnet'      : 3.0, 'sad_effnet'       : 3.0,
                                 }
    handpicked_dim_features_5 = {'mfcc_mean'  : 0.01, 'discogs_effnet_embeddings' : 5.0}

    # ---------------------------------------------------------------------------------------------
    # Running tests : Define your test parameters below for the Distance Pipeline                  |
    # ---------------------------------------------------------------------------------------------
    test_1  = ['electronic_effnet', 'instrumental_effnet']
    test_2  = ['electronic_effnet', 'acoustic_effnet']
    test_3  = ['instrumental_effnet', 'acoustic_effnet']

    # With Additional Features (vocalist, bpm)
    test_4  = ['electronic_effnet', 'bpm']
    test_5  = ['instrumental_effnet', 'bpm']
    test_6  = ['acoustic_effnet', 'bpm']

    test_7  = ['electronic_effnet', 'female_effnet']
    test_8  = ['instrumental_effnet', 'female_effnet']
    test_9  = ['acoustic_effnet', 'female_effnet']
    test_10 = ['female_effnet', 'bpm']

    # Based Around Electronic and Instrumental
    test_11 = ['electronic_effnet', 'instrumental_effnet', 'acoustic_effnet']
    test_12 = ['electronic_effnet', 'instrumental_effnet', 'female_effnet']
    test_13 = ['electronic_effnet', 'instrumental_effnet', 'bpm']

    # Based Around Electronic and Acoustic
    test_14 = ['electronic_effnet', 'acoustic_effnet', 'female_effnet']
    test_15 = ['electronic_effnet', 'acoustic_effnet', 'bpm']
    tests   = [
                test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10,
                test_11, test_12, test_13, test_14, test_15
              ]

    # Log test below.
    track_dataset_path = ''

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename='logs/handpicked_set_3.log',
        filemode='a'
    )

    logging.info("Current Settings:\n\t-Handpicked Set No.5 (COSINE TEST #1 POOLING)\n  \t-Pooled\n \t-Z-Score Normalization\n")
    for testing_track_filename in testing_tracks:
        dist_pipeline = DistPipeline(input_filename      = testing_track_filename,
                                    track_dataset_path   = track_dataset_path,
                                    numerical_dist       = DistMethods.cosine_numerical,
                                    dimensional_dist     = DistMethods.cosine_dimensional,
                                    numerical_features   = handpicked_num_features_4,
                                    dimensional_features = handpicked_dim_features_4,
                                    normalize_numerical  = z_score_normalization,
                                    pooling              = True,
                                    )
        dist_pipeline.run_pipeline(top_n=26)

    logging.info("-"*200)


