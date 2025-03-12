"""
In this file I'll be creating a wrapper for Essentia Algorithms. These algorithms are quite
simple to use relative to the models, in which you had to constantly load stuff... The main
difference is that here you're most focused on how you provide the data and how you process
the output from the algorithms.

So in the class `EssentiaAlgo` I'll just be defining a bunch of functions that wrap around the
existing Essentia Algorithms and take care of all the processing. This will make it so that all
that has to be provided is a `Track` and the method takes care of the rest.
"""

import numpy as np
import essentia.standard as es
from src.classes.track import Track

class EssentiaAlgo:
    """Just read the file docstring."""

    @staticmethod
    def get_mfcc_energy(track : Track):
        """_summary_

        Args:
            track : _description_

        Returns:
            _type_: _description_
        """

        # Prepare our needed Essentia algorithms
        mfcc           = es.MFCC()
        window         = es.Windowing(type = 'hann')
        spectrum       = es.Spectrum()


        track_mono     = track.track_mono
        mfcc_band_list = []
        frame_size     = 1024   # This is the common frame size, so I'm using it!


        for ix in range(0, len(track_mono), frame_size):

            curr_frame    = track_mono[ix:ix + frame_size]  # If the frame is shorter than
            if len(curr_frame) < frame_size:                # expected, dismiss it...
                continue

            # Process the frame: apply windowing, compute spectrum, then extract MFCC bands
            windowed_frame = window(curr_frame)
            spectrum_res   = spectrum(windowed_frame)
            mfcc_bands, _  = mfcc(spectrum_res)

            # Save results for each frame.
            mfcc_band_list.extend(mfcc_bands)


        track.features['avg_energy']  = np.mean(mfcc_band_list, axis=0)
        track.features['peak_energy'] = max(mfcc_band_list)


    @staticmethod
    def retrieve_bpm_re2013(track : Track):
        """
        ...
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(track.track_mono)

        # Update the features for our track.
        track.features['bpm'] = bpm
