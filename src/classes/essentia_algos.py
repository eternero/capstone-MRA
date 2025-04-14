"""
In this file I'll be creating a wrapper for Essentia Algorithms. These algorithms are quite
simple to use relative to the models, in which you had to constantly load stuff... The main
difference is that here you're most focused on how you provide the data and how you process
the output from the algorithms.

So in the class `EssentiaAlgo` I'll just be defining a bunch of functions that wrap around the
existing Essentia Algorithms and take care of all the processing. This will make it so that all
that has to be provided is a `Track` and the method takes care of the rest.
"""

from collections import Counter
import numpy as np
from src.utils.parallel import load_essentia_algorithm
from src.external.harmof0 import harmof0


class EssentiaAlgo:
    """Just read the file docstring."""

    @staticmethod
    def get_bpm_re2013(track_mono : np.ndarray) -> dict[str, float]:
        """
        NOTE : This can also get beats and other necessary stuff
        """
        rhythm_extractor = load_essentia_algorithm("RhythmExtractor2013", method="multifeature")
        bpm, _, _, _, _  = rhythm_extractor(track_mono)

        # Update the features for our track and return (bpm, beats)
        return {'bpm' : bpm}


    @staticmethod
    def get_energy(track_mono : np.ndarray) -> dict[str, float]:
        """Get the track energy using the Essentia Energy Algorithm."""
        energy_extractor         = load_essentia_algorithm("Energy")
        track_energy             = energy_extractor(track_mono)
        return {'energy' : track_energy}


    @staticmethod
    def get_loudness_ebu_r128(track_path : str):
        """..."""
        audio_loader       = load_essentia_algorithm("AudioLoader", filename=track_path)
        loudness_extractor = load_essentia_algorithm("LoudnessEBUR128")

        stereo_signal, _, _, _, _, _              = audio_loader()
        _, _, integrated_loudness, loudness_range = loudness_extractor(stereo_signal)
        return {'integrated_loudness' : integrated_loudness, 'loudness_range' : loudness_range}


    @staticmethod
    def harmonic_f0(track_mono : np.ndarray, device : str = 'mps'):
        """Not exactly an Essentia Algorithm, but this is the replacement (and improvement)
        over CREPE. We're using HarmoF0 for pitch estimation and to acquire significant features
        based on the track's pitch.
        """

        pitch_tracker              = harmof0.PitchTracker(device=device)
        _, freq, _, activation_map = pitch_tracker.pred(track_mono, 16000)

        # First, we compute the weighted histogram from the activation map.
        activation_vec_length   = activation_map.shape[1]
        weighted_histogram      = np.zeros(activation_vec_length)

        for activation_vec in activation_map:
            weighted_histogram += activation_vec

        # Once we're done summing all the activation vectors, normalize the histogram.
        weighted_histogram     /= np.sum(weighted_histogram)

        # Now acquire the statistical features for the frequency and save em' all to track.
        features = {
                'pitch_hist' : list(weighted_histogram),
                'pitch_mean' : np.mean(freq, axis=0),
                'pitch_var'  : np.var(freq, axis=0)
        }
        return features

    @staticmethod
    def mfcc_renewed(track_mono  : np.ndarray,
                     frame_size  : int = 2048,
                     num_bands   : int = 40,
                     num_coeff   : int = 13):
        """This method uses Essentia's MFCC Algorithm to create meaningful features.

        The parameters used for this mainly regard the MFCC function, for example we can change
        the number of bands and mel-frequency cepstrum coefficients (mfcc) returned by the MFCC
        Algorithm.

        It is important to mention that we're using a 50% overlap for windows as seen with the usage
        of `half_frame` steps as we iterate. This is done following the procedures of Mahieux et al. in

            _"Proceedings of the 12th International Society for Music Information Retrieval Conference"_

        Args:
            track_path : The string pointing to the path of the track that we'll process.
            frame_size : The frame size used for our Windows & acquisition of Spectrum. Defaults to 2048
            num_bands  : The number of mel bands to be returned by MFCC(). Defaults to 40
            num_coeff  : The number of mfccs to be returned by MFCC(). Defaults to 13
        """

        # Define basics
        half_frame      = frame_size // 2
        band_list       = []
        mfcc_list       = []


        # Define Essentia Stuff
        windowing      = load_essentia_algorithm("Windowing",type='hann')
        spec_extractor = load_essentia_algorithm("Spectrum" , size=frame_size)
        mfcc_extractor = load_essentia_algorithm("MFCC", numberBands = num_bands,
                                                numberCoefficients = num_coeff)


        for ix in range(0, len(track_mono), half_frame):    # Use half-frames as our step so that we can
            curr_frame = track_mono[ix : ix + frame_size]   # Have a 50% overlap fr all of the frames.

            # Don't process incomplete frames.
            if len(curr_frame) < frame_size:
                continue

            window_frame = windowing(curr_frame)
            spectrum     = spec_extractor(window_frame)
            bands, mfcc  = mfcc_extractor(spectrum)

            band_list.append(bands)
            mfcc_list.append(mfcc)


        # Once we're out of the loop, gather all our features.
        bands_array = np.vstack(band_list)
        mfcc_array  = np.vstack(mfcc_list)


        features = {
            'mfcc_mean' : list(np.mean(mfcc_array, axis=0)),
            'mfcc_std'  : list(np.std(mfcc_array, axis=0)),

            'band_mean' : list(np.mean(bands_array, axis=0)),
            'band_std'  : list(np.std(bands_array, axis=0)),
        }

        return features


    @staticmethod
    def el_monstruo(track_mono : np.ndarray):
        """Extracts the following features :
            - Most Common Chords
            - Most Common Keys
            - MFCC Peak Energy
            - MFCC Average Energy

        Furthermore, since the MFCC is acquired in this method,
        it can be modified for additional analysis of the tracks.
        """

        # Define constants and extractors.
        frame_size     = 1024
        key_extract    = load_essentia_algorithm("Key")
        mfcc           = load_essentia_algorithm("MFCC")
        hpcp           = load_essentia_algorithm("HPCP")
        window         = load_essentia_algorithm("Windowing",type='hann')
        spectrum       = load_essentia_algorithm("Spectrum" , size=frame_size)
        dissonance     = load_essentia_algorithm("Dissonance")
        spec_peaks     = load_essentia_algorithm("SpectralPeaks")
        rollof_calc    = load_essentia_algorithm("RollOff")
        tristimulus    = load_essentia_algorithm("Tristimulus")
        chord_detect   = load_essentia_algorithm("ChordsDetection")
        pitch_salience = load_essentia_algorithm("PitchSalience")

        """
        frame_size     = 1024
        key_extract    = es.Key()
        mfcc           = es.MFCC()
        hpcp           = es.HPCP()
        window         = es.Windowing(type='hann')
        spectrum       = es.Spectrum(size=frame_size)
        dissonance     = es.Dissonance()
        spec_peaks     = es.SpectralPeaks()
        rollof_calc    = es.RollOff()
        tristimulus    = es.Tristimulus()
        chord_detect   = es.ChordsDetection()
        pitch_salience = es.PitchSalience()
        """

        # Define result lists.
        pcp_list       = []
        key_list       = []
        diss_list      = []
        pitch_list     = []
        timbre_list    = []
        rollof_list    = []
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
            pitch_profile    = hpcp(freq, magnitudes)
            timbre           = tristimulus(freq, magnitudes)
            key, scale, _, _ = key_extract(pitch_profile)
            pitch            = pitch_salience(spectrum_res)
            rollof           = rollof_calc(spectrum_res)


            # Skip this frame if there are no spectral peaks or if it's empty.
            if freq.size == 0 or magnitudes.size == 0:
                continue

            # Sort arrays in ascending order using NumPy
            order             = np.argsort(freq)
            sorted_freq       = freq[order]
            sorted_magnitudes = magnitudes[order]
            diss              = dissonance(sorted_freq, sorted_magnitudes)

            # Append acquired values to lists
            pcp_list.append(pitch_profile)
            key_list.append((key, scale))       # TODO : Refactor to a "".join(list) for more efficiency
            diss_list.append(diss)
            pitch_list.append(pitch)
            timbre_list.append(timbre)
            rollof_list.append(rollof)
            mfcc_band_list.extend(mfcc_bands)


        # Gather the top chords and keys...
        chords, _      = chord_detect(pcp_list)
        top_chords     = Counter(chords).most_common(5)
        top_keys       = Counter(key_list).most_common(5)

        # Save track features
        features = {
            'most_common_chords' : [chord for chord, _ in top_chords],
            'most_common_keys'   : [f"{key}_{scale}" for ((key, scale), _) in top_keys],
            'mfcc_peak_energy'   : max(mfcc_band_list),
            'mfcc_avg_energy'    : np.mean(mfcc_band_list, axis=0),
            'pitch_salience'     : np.mean(pitch_list,     axis=0),
            'avg_dissonance'     : np.mean(diss_list,      axis=0),
            'avg_rollof'         : np.mean(rollof_list,    axis=0),
            'tristimulus'        : tuple(np.mean(np.array(timbre_list), axis=0))
            # Btw tristimulus and timbre are interchangeable. But the 'timbre' feature is reserved for models
        }

        return features
