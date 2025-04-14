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
    def get_spectral_centroid_time(track_mono : np.ndarray):
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
        window   = es.Windowing(type='hann') 
        spectrum = es.Spectrum()             
        centroid = es.SpectralCentroidTime() 

        frame_size = 1024
        centroids = []

        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed    = window(frame)
            spec        = spectrum(windowed)
            centroids.append(centroid(spec))

        # Saving statistics
        centroids = np.array(centroids)
        features = {
            'spectral_centroid_mean':   np.mean(centroids),
            'spectral_centroid_std' :   np.std(centroids),
            'spectral_centroid_max' :   np.max(centroids),
        }
        return features

    @staticmethod
    def get_spectral_rolloff(track_mono : np.ndarray):
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
        window      = es.Windowing(type='hann') 
        spectrum    = es.Spectrum()             
        rolloff     = es.RollOff()              

        frame_size   = 1024
        rolloffs     = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec = spectrum(windowed)
            rolloffs.append(rolloff(spec))
        
        rolloffs = np.array(rolloffs)
        feature = {    
            'rolloff_mean' : np.mean(rolloffs),
            'rolloff_std' : np.std(rolloffs),
            'rolloff_max' : np.max(rolloffs),
        }
        return feature

    @staticmethod
    def get_spectral_contrast(track_mono : np.ndarray):
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
        frame_size  = 1024
        window      = es.Windowing(type='hann') 
        spectrum    = es.Spectrum()             
        # Adding the frame_size parameter since it defaults to 2048 and gives an error
        contrast    = es.SpectralContrast(frameSize=frame_size)              


        contrasts   = []
        valleys     = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if(len(frame) < frame_size):
                continue

            windowed    = window(frame)
            spec        = spectrum(windowed)
            contrast_values, contrast_valleys = contrast(spec)
            contrasts.append(contrast_values)
            valleys.append(contrast_valleys)

        contrasts = np.array(contrasts)
        valleys   = np.array(valleys)
        feature = {
            'spectral_contrast_mean' : np.mean(contrasts, axis=0),
            'spectral_contrast_std'  : np.std(contrasts, axis=0),
            'spectral_valley_mean'   : np.mean(valleys, axis=0),
            'spectral_valley_std'    : np.std(valleys, axis=0)
        }
        return feature

    @staticmethod
    def get_hfc(track_mono : np.ndarray):
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
        window   = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        hfc      = es.HFC()

        frame_size = 1024
        hfc_vals   = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            hfc_vals.append(hfc(spec))

        hfc_vals = np.array(hfc_vals)
        feature = {
            'hfc_mean' : np.mean(hfc_vals),
            'hfc_std'  : np.std(hfc_vals)
        }
        return feature

    @staticmethod
    def get_flux(track_mono : np.ndarray):
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
        window   = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        flux     = es.Flux()

        frame_size = 1024
        flux_vals  = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            flux_values = flux(spec)
            flux_vals.append(flux_values)


        flux_vals = np.array(flux_vals)
        feature = {
            'flux_mean' : np.mean(flux_vals),
            'flux_std'  : np.std(flux_vals)
        }
        return feature
    
    @staticmethod
    def get_flatness_db(track_mono : np.ndarray):
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
        window    = es.Windowing(type='hann')
        spectrum  = es.Spectrum()
        flatness  = es.FlatnessDB()

        frame_size = 1024
        flatness_vals = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i:i+frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            flatness_vals.append(flatness(spec))

        flatness_vals = np.array(flatness_vals)
        feature = {
            'flatness_db_mean'  : np.mean(flatness_vals),
            'flatness_db_std'   : np.std(flatness_vals)
        }
        return feature

    @staticmethod
    def get_energy_band_ratio(track_mono : np.ndarray):
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


        window     = es.Windowing(type='hann')
        spectrum   = es.Spectrum()

        frame_size = 1024

        band_ratios = []
        freq_bands  = [(0, 60), (60, 250),
                      (250, 500), (500, 2000), 
                      (2000, 4000), (4000, 6000),
                      (6000, 22050)]
        
        band_names = ['sub_bass', 'bass', 
                      'lower_midrange', 'midrange', 
                      'higher_midrange', 'presence', 
                      'brilliance']

        # Create an instance per band
        energy_band = [
            es.EnergyBandRatio(
                sampleRate=44100,
                startFrequency=start,
                stopFrequency=stop
            )
            for (start, stop) in freq_bands]


        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            
            ratios = [band_algo(spec) for band_algo in energy_band]
            band_ratios.append(ratios)

        ratios_array = np.array(band_ratios)
        features = {}
        for idx, name in enumerate(band_names):
            features[f'energy_ratio_{name}_mean'] = np.nanmean(ratios_array[:, idx])
            features[f'energy_ratio_{name}_std']  = np.nanstd(ratios_array[:, idx])

        return features
    
    @staticmethod
    def get_spectral_peaks(track_mono : np.ndarray):
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
        window      = es.Windowing(type='blackmanharris92')
        spectrum    = es.Spectrum()
        spec_peaks  = es.SpectralPeaks()

        frame_size = 1024
        peak_counts, freqs, mags = [], [], []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)

            frequency, magnitude = spec_peaks(spec)
            peak_counts.append(len(frequency))
            if len(frequency) > 0:
                freqs.extend(frequency)
                mags.extend(magnitude)

        if freqs:
            features = {
                'spectral_peaks_avg_freq': np.mean(freqs),
                'spectral_peaks_avg_mag' : np.mean(mags),
                'spectral_peaks_count'   : np.mean(peak_counts)
            }
        else:
            features = {
                'spectral_peaks_avg_freq': 0,
                'spectral_peaks_avg_mag' : 0,
                'spectral_peaks_count'   : 0
            }
        return features
    
    @staticmethod
    def get_gfcc(track_mono : np.ndarray):
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
        window   = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        gfcc     = es.GFCC()

        frame_size  = 1024
        erb_bands   = []
        gfcc_coeffs = []

        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec = spectrum(windowed)
            bands, coeffs = gfcc(spec)
            erb_bands.append(bands)
            gfcc_coeffs.append(coeffs)

        erb_bands   = np.array(erb_bands)
        gfcc_coeffs = np.array(gfcc_coeffs)

        feature= {
            'erb_bands_mean': np.mean(erb_bands, axis=0).tolist(),
            'erb_bands_std' : np.std(erb_bands, axis=0).tolist(),
            'gfcc_mean'      : np.mean(gfcc_coeffs, axis=0).tolist(),
            'gfcc_peak'      : np.max(gfcc_coeffs, axis=0).tolist()
        }

        return feature
    
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
