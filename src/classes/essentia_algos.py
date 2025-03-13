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
from essentia.standard import MonoLoader
from collections import Counter

class EssentiaAlgo:
    """Just read the file docstring."""

    @staticmethod
    def get_bpm_re2013(track : Track) -> tuple[float, list[float]]:
        """
        ...
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, _, _, _ = rhythm_extractor(track.track_mono)

        # Update the features for our track and return (bpm, beats)
        track.features['bpm']   = bpm
        return bpm, beats


    @staticmethod
    def get_energy(track : Track):
        """Get the track energy using the Essentia Energy Algorithm."""
        track.features['energy'] = es.Energy()(track.track_mono)


    @staticmethod
    def get_loudness(track : Track):
        """..."""
        track.features['loudness'] = es.Loudness()(track.track_mono)


    @staticmethod
    def get_time_signature(track : Track):
        """This method is used to acquire the time signature of a track. It must be mentioned
        however that the Essentia Meter method is Experimental... and not advised to be used.
        Oh well.
        """
        # Acquire the Loudness and Loudness Band Ratio
        _, beats                           = EssentiaAlgo.get_bpm_re2013(track)
        beat_loudness, loudness_band_ratio = es.BeatsLoudness(beats = beats)(track.track_mono)

        # Acauire the beatogram and save the time signature.
        beatogram                          = es.Beatogram()(beat_loudness, loudness_band_ratio)
        time_signature                     = es.Meter()(beatogram)
        track.features['time_signature']   = time_signature


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

            curr_frame     = track_mono[ix:ix + frame_size]  # If the frame is shorter than
            if len(curr_frame) < frame_size:                 # expected, dismiss it...
                continue

            # Process the frame: apply windowing, compute spectrum, then extract MFCC bands
            windowed_frame = window(curr_frame)
            spectrum_res   = spectrum(windowed_frame)
            mfcc_bands, _  = mfcc(spectrum_res)

            # Save results for each frame.
            mfcc_band_list.extend(mfcc_bands)

        track.features['mfcc_avg_energy']  = np.mean(mfcc_band_list, axis=0)
        track.features['mfcc_peak_energy'] = max(mfcc_band_list)


    @staticmethod
    def el_monstruo(track : Track):
        """Extracts the following features :
            - Most Common Chords
            - Most Common Keys
            - MFCC Peak Energy
            - MFCC Average Energy

        Furthermore, since the MFCC is acquired in this method,
        it can be modified for additional analysis of the tracks.
        """
        key_extract    = es.Key()
        mfcc           = es.MFCC()
        hpcp           = es.HPCP()
        window         = es.Windowing(type = 'hann')
        spectrum       = es.Spectrum()
        spec_peaks     = es.SpectralPeaks()
        chord_detect   = es.ChordsDetection()

        track_mono     = track.track_mono
        frame_size     = 1024

        # Apparently I'm supposed to gather these...
        pcp_list       = []
        key_list       = []
        mfcc_band_list = []

        for ix in range(0, len(track_mono), frame_size):

            curr_frame     = track_mono[ix:ix + frame_size]  # If the frame is shorter than
            if len(curr_frame) < frame_size:                 # expected, dismiss it...
                continue

            # Process the frames of the track.
            windowed_frame   = window(curr_frame)
            spectrum_res     = spectrum(windowed_frame)
            mfcc_bands, _    = mfcc(spectrum_res)

            freq, magnitudes = spec_peaks(spectrum_res)
            pcp              = hpcp(freq, magnitudes)
            key, scale, _, _ = key_extract(pcp)

            pcp_list.append(pcp)
            key_list.append((key, scale))
            mfcc_band_list.extend(mfcc_bands)


        # Gather the top chords and keys...
        chords_list, _ = chord_detect(pcp_list)
        chord_counter  = Counter(chords_list)
        key_counter    = Counter(key_list)
        top_chords     = chord_counter.most_common(5)
        top_keys       = key_counter.most_common(5)

        # Save track features
        track.features['most_common_chords'] = [chord for chord, _ in top_chords]
        track.features['most_common_keys']   = [f"{key}_{scale}" for ((key, scale), _) in top_keys]
        track.features['mfcc_peak_energy']   = max(mfcc_band_list)
        track.features['mfcc_avg_energy']    = np.mean(mfcc_band_list, axis=0)

