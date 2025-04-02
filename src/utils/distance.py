"""
This file will contain all methods that serve as helpers for the procedure of acquiring distances
between features in our dataset. This not only includes the distance methods themselves, but other
methods which instead process our data so that they can be compared with distance metrics.
"""

from typing import Callable
import numpy as np
from src.classes.track import Track


# -------------------------------------------------------------------------------------------------
# Processing Methods
# -------------------------------------------------------------------------------------------------

def build_weighted_pitch_profile(activation_matrix):
    """Builds a weighted Pitch Profile (Histogram) by summing the activation vectors for
    all frames in a track. The Activation Matrix is acquired from running the HarmoF0 Pitch Tracker.

    Args:
        activation_matrix : The activation matrix returned by the Pitch Estimation Model.
                            In the case of HarmoF0, this should be of shape (352,)
    """
    # Initialize a n-dimensional array for the histogram
    matrix_size   = activation_matrix.shape[1]
    weighted_hist = np.zeros(matrix_size)

    for activation_bin in activation_matrix:
        weighted_hist += activation_bin  # element-wise addition

    # Normalize the histogram to form a probability distribution
    weighted_hist /= np.sum(weighted_hist)
    return weighted_hist


# -------------------------------------------------------------------------------------------------
# Distance Methods
# -------------------------------------------------------------------------------------------------

def dist_helper(input_track : Track, track_list : list[Track], dist_func : Callable, top_ix : int = 15):
    """..."""
    result_dict = {}
    print(f"Selected Track : {input_track.track_name}")

    for curr_track in track_list:
        track_name = curr_track.track_name
        dist_score = dist_func(input_track.track_hist, curr_track.track_hist)
        result_dict[track_name] = dist_score

    # Sort the output and print the `top_ix` closest scores.
    # NOTE : This is to be removed, as it is only for testing at the moment.
    output = dict(sorted(result_dict.items(), key=lambda item: item[1]))
    for ix, items in enumerate(output.items()):
        if ix >= top_ix:
            break
        key, val = items
        print(key, val)

    return output


# NOTE : Have to modify all of this depending on how I end up dealing with features.
def minkowski_dist(feature_list : list[str], input_track : Track, comparison_track : Track, p : int = 2):
    """
    Essentia the Ln Norm function. For this, we have the following:
        * p = 1 : Manhattan Distance
        * p = 2 : Euclidean Distance

    By default, this will be set to Euclidean (p=2), however it must be noted that
    throughout testing, both Manhattan and Euclidean yielded amazing results. Other
    methods such as Mahalanobis and Cosine distance were not as good.

    NOTE : There are more distance methods yet to be tested such as Earth's Mover.
    """
    dist_score = 0
    for feature in feature_list:

        input_feat   = input_track[feature]
        compare_feat = comparison_track[feature]
        dist_score   += (input_feat - compare_feat)**p

    return float(dist_score)

def euclidean_distance_np(hist1, hist2):
    """Euclidean distance for Numpy Arrays"""
    return np.linalg.norm(hist1 - hist2)

