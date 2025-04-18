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
import essentia.standard as es


class EssentiaAlgo:
    """Just read the file docstring."""

    #-------------------------------------------------------------------------------------------------
    #   Shared utils for spectral algorithms
    #-------------------------------------------------------------------------------------------------
    FRAME_SIZE = 1024
    HOP_SIZE   = 1024

    _window   = es.Windowing(type="hann")
    _spectrum = es.Spectrum()

    @classmethod
    def _spectral_frames(cls, track_mono: np.ndarray):
        """
        Yield for spectrum frames — To remove redundancies
        """
        n = len(track_mono)
        for start in range(0, n - cls.FRAME_SIZE, cls.HOP_SIZE):
            frame = track_mono[start : start + cls.FRAME_SIZE]
            yield cls._spectrum(cls._window(frame))

    @staticmethod
    def _stats(values: np.ndarray, name: str, include_max: bool=False) -> dict[str, float]:
        """
        Return mean / std / (max) for a vector with its keys
        """
        features: dict[str, float] = {
            f"{name}_mean": float(np.mean(values)),
            f"{name}_std": float(np.std(values))
        }
        if include_max:
            features[f"{name}_max"] = float(np.max(values))
        return features

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


    @classmethod
    def get_spectral_centroid_time(cls, track_mono : np.ndarray):
        """
        Computes the spectral centroid of the audio over time.

        The spectral centroid indicates the "brightness" of the sound,
        representing the center of mass of the spectrum. Higher values suggest 
        brighter, sharper sounds; lower values indicates darker, bass-heavy sounds

        Usefulness:
            Helps distinguish bright, sharp tracks (like electronic or pop) from darker,
            bass-heavy sounds (like hip-hop or reggaeton), contributing to mood and genre recognition.
        
        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - spectral_centroid_mean: Mean spectral centroid (Hz).
            - spectral_centroid_std: Standard deviation of spectral centroid.
            - spectral_centroid_max: Maximum spectral centroid value.
        """         
        algorithm = es.SpectralCentroidTime() 
        vals      = np.array([algorithm(spectrum) for spectrum in cls._spectral_frames(track_mono)])
        return cls._stats(vals, "spectral_centroid", include_max=True)

    @classmethod
    def get_spectral_rolloff(cls, track_mono : np.ndarray):
        """
        Computes the spectral rolloff  of a track

        Spectral rolloff calculates the frequency below a given percentage (default 85%)
        of the total spectral energy. Higher values suggest bright, noisy sounds; 
        lower values point to bass-heavy tracks
    
        Usefulness:
            Shows the energy distribution under a certain point, pretty useful when a song 
            is dark but it has some beats that makes it appear that is kinda bright.
            It shows where the energy is mostly under that threshold.

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - rolloff_mean: Mean roll-off frequency (Hz).
            - rolloff_std: Standard deviation of roll-off frequency.
            - rolloff_max: Maximum roll-off frequency.
        """
        algorithm = es.RollOff()              
        values    = np.array([algorithm(spectrum) for spectrum in cls._spectral_frames(track_mono)])
        return cls._stats(values, "rolloff", include_max=True)

    @classmethod
    def get_spectral_contrast(cls, track_mono : np.ndarray):
        """
        Computes Spectral Contrast across several frequency bands
        
        Spectral contrast measures the difference betweem peaks and valleys
        in the frequency spectrum. Higher contrast indicates rich, bright harmonic
        structures (good for differentiating instruments).

        Usefulness:
            Helps separate smooth, clean genres (like classical) from noisy, dense textures
            (like distorted rock or heavily compressed electronic music).

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - spectral_contrast_mean: Mean spectral contrast per band.
            - spectral_contrast_std: Standard deviation of spectral contrast per band.
            - spectral_valley_mean: Mean spectral valley per band
            - spectral_valley_std: Standard deviation of the valley

            It outputs an array, each value corresponds to the contrast of a specific 
            frequency band, it captures the `texture` per frequency region. Essentia
            defaults to 6 frequency bands.
        """    
        # Adding the frame_size parameter since it defaults to 2048 and gives an error
        algorithm = es.SpectralContrast(frameSize=cls.FRAME_SIZE)
        contrasts, valleys = [], []
        for spectrum in cls._spectral_frames(track_mono):
            contrast_values, contrast_valleys = algorithm(spectrum)
            contrasts.append(contrast_values)
            valleys.append(contrast_valleys)

        contrasts = np.array(contrasts)
        valleys   = np.array(valleys)
        return {
            'spectral_contrast_mean' : np.mean(contrasts, axis=0),
            'spectral_contrast_std'  : np.std(contrasts, axis=0),
            'spectral_valley_mean'   : np.mean(valleys, axis=0),
            'spectral_valley_std'    : np.std(valleys, axis=0)
        }


    @classmethod
    def get_hfc(cls, track_mono : np.ndarray):
        """
        Compute the High Frequency Content of the track.

        HFC emphasizes high-frequency energy, useful for detecting bright or sharp tracks.
        Higher values generally correspond to energetic or high-frequency-heavy content.

        Usefulness:
            Helps highlight energetic, bright tracks, also read that it complements Flux to confirm
            that detected frequencies are high-frequency events.        

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - hfc_mean: Mean high-frequency content.
            - hfc_std: Standard deviation of high-frequency content.

        """
        algorithm = es.HFC()
        values    = np.array([algorithm(spectrum) for spectrum in cls._spectral_frames(track_mono)])
        return cls._stats(values, "hfc")
         

    @classmethod
    def get_flux(cls, track_mono : np.ndarray):
        """
        Compute spectral flux over the track.
        
        Spectral flux measures the rate of change in the power spectrum between frames.
        High flux indicates energetic, rapidly changing tracks; low flux suggests stability.

        Usefulness:
            Useful for identifying energetic, eventful tracks with lots of changes
            versus stable, smooth tracks like ambient or continuous pads.

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - flux_mean: Mean spectral flux.
            - flux_std: Standard deviation of spectral flux.
        """
        algorithm = es.Flux()
        values    = np.array([algorithm(spectrum) for spectrum in cls._spectral_frames(track_mono)])
        return cls._stats(values, "flux")
    
    @classmethod
    def get_flatness_db(cls, track_mono : np.ndarray):
        """
        Compute spectral flatness in dB.
        
        Spectral flatness measures how noise-like a sound is.
        High flatness (~1.0) indicates noise or chaotic spectrum;
        low flatness (~0.0) suggests tonal, harmonic structure.

        Usefulness:
            Good for distinguishing between harmonic, melodic content and noisy,
            chaotic textures like percussive or distorted sounds.
        
        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - flatness_db_mean: Mean spectral flatness.
            - flatness_db_std: Standard deviation of spectral flatness.

        """
        algorithm = es.FlatnessDB()
        values    = np.array([algorithm(spectrum) for spectrum in cls._spectral_frames(track_mono)])
        return cls._stats(values, "flatness_db")

    @classmethod
    def get_energy_band_ratio(cls, track_mono : np.ndarray):
        """
        Compute energy band ratios for defined frequencies.
        
        Provides targeted analysis of energy distribution to help characterize 
        bass-heavy, mid-focused, or bright tracks. 

        Usefulness:
            Since it separates the spectrum in bands it helps to classify
            tracks based on its energy focus since we can see where which band
            has more energy. bass-heavy, midrange-rich, or bright.

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - energy_band_ratio_{freq_band_name}_mean: Average frequency respective to the band
            - energy_band_ratio_{freq_band_name}_std: Standard deviation from each respective band

            Output is manually split into multiple "outputs", they represent a specific frequency range
        """
        freq_bands = [(0, 60), (60, 250),
                      (250, 500), (500, 2000), 
                      (2000, 4000), (4000, 6000),
                      (6000, 22050)]
        
        band_names = ['sub_bass', 'bass', 
                      'lower_midrange', 'midrange', 
                      'higher_midrange', 'presence', 
                      'brilliance']

        # Create an instance per band
        band_algorithms = [
            es.EnergyBandRatio(sampleRate=44100, startFrequency=start, stopFrequency=stop)
            for (start, stop) in freq_bands
        ]
        frame_ratios = [
            [algo(spec) for algo in band_algorithms] for spec in cls._spectral_frames(track_mono)
        ]
        ratios    = np.array(frame_ratios)
        frequency = {}
        for i, name in enumerate(band_names):
            frequency[f"energy_ratio_{name}_mean"] = float(np.nanmean(ratios[:, i]))
            frequency[f"energy_ratio_{name}_std"]  = float(np.nanstd(ratios[:, i]))
        
        return frequency
    
    @classmethod
    def get_spectral_peaks(cls, track_mono : np.ndarray):
        """
        Extract spectral peaks (frequencies and magnitudes).

        Useful for analyzing tonal complexity.

        Usefulness:
            Good for differentiating tracks that are complex, good for understanding
            overall tonal density.

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - spectral_peaks_avg_freq: Average frequency of spectral peaks (Hz).
            - spectral_peaks_avg_mag: Average magnitude of spectral peaks.
            - spectral_peaks_count: Average number of peaks per frame.
        """
        window_bh = es.Windowing(type="blackmanharris92")
        algorithm = es.SpectralPeaks()
        spectrum  = cls._spectrum
        peak_counts, freqs, mags = [], [], []

        n = len(track_mono)
        for start in range(0, n - cls.FRAME_SIZE + 1, cls.HOP_SIZE):
            frame = track_mono[start : start + cls.FRAME_SIZE]
            spec = spectrum(window_bh(frame))
            freq_vals, mags_vals = algorithm(spec)
            peak_counts.append(len(freq_vals))
            freqs.extend(freq_vals)
            mags.extend(mags_vals)

        return {
            'spectral_peaks_avg_freq': np.mean(freqs),
            'spectral_peaks_avg_mag' : np.mean(mags),
            'spectral_peaks_count'   : np.mean(peak_counts)
        }
    
    @classmethod
    def get_gfcc(cls, track_mono : np.ndarray):
        """
        Computes the Gammatone Frequency Cepstral Coefficients (GFCC) and ERB band energies.

        GFCC captures timbral details better in noisy enviroments, good when a song is 
        poorly mastered or is noisy, while the ERB bands provide a snapshot the energy distribution.
        Gammatone Filters are modeled after the `human ear's basilar membrane response`, 
        which in theory it does better at isolating relevant frequency information.

        Usefulness:
            Good for extracting a tracks timbral features from noisy songs — helps 
            the system to distinguish the tracks more accurately.

        Inputs:
            track_mono (np.ndarray): Mono audio signal of the track

        Outputs:
            - gfcc_avg: Average GFCC coefficients.
            - gfcc_peak: Peak GFCC coefficients.
            - erb_bands_mean: Mean energy in ERB bands.
            - erb_bands_std: Standard deviation of energy in ERB bands.

            Output is an array, for gfcc is a timbral descriptor extracted from the Gammatone filterbank; 
            the ERB arrays correspond to energy in one of the bands. 

            NOTE - erb array is 40 items, gfcc array 13 
        """
        algorithm = es.GFCC()
        erb_bands, gfcc_coeffs = [], []

        for spec in cls._spectral_frames(track_mono):
            erb, coeff = algorithm(spec)
            erb_bands.append(erb)
            gfcc_coeffs.append(coeff)

        erb_bands   = np.array(erb_bands)
        gfcc_coeffs = np.array(gfcc_coeffs)

        return {
            'erb_bands_mean' : np.mean(erb_bands, axis=0).tolist(),
            'erb_bands_std'  : np.std(erb_bands, axis=0).tolist(),
            'gfcc_mean'      : np.mean(gfcc_coeffs, axis=0).tolist(),
            'gfcc_peak'      : np.max(gfcc_coeffs, axis=0).tolist()
        }
    
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
