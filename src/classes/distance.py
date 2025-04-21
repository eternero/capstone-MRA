"""Everything that has to do with distance."""
import ast
import logging
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='distance_ind.log',
    filemode='a'
)


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

        distance = 0
        for feature, weight in numerical_features.items():
            input_feat = input_track[feature]
            comp_feat  = comp_track[feature]

            distance   += ((input_feat - comp_feat)**p) * weight

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

        input_features = [input_track[feat] * weight for feat, weight in numerical_features.items()]
        comp_features  = [comp_track[feat] * weight for feat, weight in numerical_features.items()]

        cosine_dot     = np.dot(input_features, comp_features)
        cosine_norm    = (np.linalg.norm(input_features)) * (np.linalg.norm(comp_features))
        cosine_sim     = cosine_dot / cosine_norm

        return float(cosine_sim)

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
            input_feat = np.array(ast.literal_eval(input_track[feature]))
            comp_feat  = np.array(ast.literal_eval(comp_track[feature]))

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
            input_feature = np.array(ast.literal_eval(input_track[feature]))
            comp_feature  = np.array(ast.literal_eval(comp_track[feature]))

            cosine_dot    = np.dot(input_feature, comp_feature)
            cosine_norm   = (np.linalg.norm(input_feature)) * (np.linalg.norm(comp_feature))
            dist_score    += (cosine_dot / cosine_norm) * weight

        return float(dist_score)


class DistPipeline:
    """
    Runs the entire procedure of acquiring the distance between an input track and a track dataset.
    Handles normalization of dataset and

    Args:
        TODO
    """

    def __init__(self,
                 input_filename        : str,
                 track_dataset_path    : str,
                 numerical_dist        : Callable,
                 dimensional_dist      : Callable,
                 numerical_features    : dict[str, float] = None,
                 dimensional_features  : dict[str, float] = None,
                 expand_dimensional    : bool     = False,
                 normalize_numerical   : Callable = None,
                 normalize_dimensional : Callable = None,
                ):

        self.input_filename        = input_filename
        self.track_dataset_path    = track_dataset_path
        self.numerical_features    = numerical_features
        self.dimensional_features  = dimensional_features
        self.numerical_dist        = numerical_dist
        self.dimensional_dist      = dimensional_dist
        self.normalize_numerical   = normalize_numerical
        self.normalize_dimensional = normalize_dimensional


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


        if self.normalize_numerical:
            num_ft_list                    = list(self.numerical_features.keys())
            self.mod_track_df[num_ft_list] = normalize_numerical(self.mod_track_df[num_ft_list])

        if self.normalize_dimensional:
            dim_ft_list                    = self.dimensional_features.keys()
            self.mod_track_df[dim_ft_list] = normalize_numerical(self.mod_track_df[dim_ft_list])


        # Once we're done handling normalization, get our input track.
        self.input_df : pd.DataFrame = self.mod_track_df.loc[self.input_filename]


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


    def run_pipeline(self):
        """TODO"""

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
            if ix >= 21:
                break
            key, val = items
            logging.info("%d. %s - %f", ix, key, val)
        logging.info("\n"*2)


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
                                 # 'party_effnet': 1.0, 'sad_effnet': 1.0,
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
    #   - Addind the `avg_dissonance` makes it a bit better surprisingly. Still not good, but somewhat
    #     more considerable.
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
    handpicked_num_norm_2     = z_score_normalization
    handpicked_num_features_2 = { 'electronic_effnet' : 1.0, 'instrumental_effnet' : 1.0, 'acoustic_effnet' : 0.8,
                                  'female_effnet' : 1.0, 'bpm' : 0.65 }
    handpicked_dim_features_2 = {'mfcc_mean' : 0.06, 'mfcc_std' : 0.15}     # Not adding tristimulus yet.

    # ---------------------------------------------------------------------------------------------
    # Running tests : Define your test parameters below for the Distance Pipeline                  |
    # ---------------------------------------------------------------------------------------------
    logging.info("Current Settings:\n  \t-TEST MESSAGE\n\t-Euclidean for Both Distances\n \t-No Normalization\n")

    track_dataset_path = ''
    for testing_track_filename in testing_tracks:

        dist_pipeline = DistPipeline(input_filename      = testing_track_filename,
                                    track_dataset_path   = track_dataset_path,
                                    numerical_dist       = DistMethods.euclidean_numerical,
                                    dimensional_dist     = DistMethods.euclidean_dimensional,
                                    numerical_features   = None,
                                    dimensional_features = None,
                                    normalize_numerical  = None
                                    )
        dist_pipeline.run_pipeline()

    logging.info("-"*200)
