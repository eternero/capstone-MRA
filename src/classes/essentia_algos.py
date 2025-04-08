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
import essentia.standard as _es  
from src.classes.track import Track
from collections import Counter

# To shut the linter off 
from typing import Any
es: Any = _es
class EssentiaAlgo:
    """Just read the file docstring."""

    @staticmethod
    def get_bpm_re2013(track : Track) -> tuple[float, list[float]]:
        """
        NOTE : This can also get beats and other necessary stuff
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _  = rhythm_extractor(track.track_mono_44)

        # Update the features for our track and return (bpm, beats)
        track.features['bpm'] = bpm


    @staticmethod
    def get_energy(track : Track):
        """Get the track energy using the Essentia Energy Algorithm."""
        track.features['energy'] = es.Energy()(track.track_mono_44)


    @staticmethod
    def get_intensity(track : Track):
        """NOTE : Quality: outdated (non-reliable, poor accuracy). Yea, it fucking sucks"""
        track.features['intensity'] = es.Intensity()(track.track_mono_44)


    @staticmethod
    def get_loudness_ebu_r128(track : Track):
        """..."""
        stereo_signal, _, _, _, _, _              = es.AudioLoader(filename=track.track_path)()
        _, _, integrated_loudness, loudness_range = es.LoudnessEBUR128()(stereo_signal)
        track.features['integrated_loudness']     = integrated_loudness
        track.features['loudness_range']          = loudness_range



    @staticmethod
    def get_time_signature(track : Track):
        """This method is used to acquire the time signature of a track. It must be mentioned
        however that the Essentia Meter method is Experimental... and not advised to be used.
        Oh well.
        """
        # Acquire the Loudness and Loudness Band Ratio
        _, beats, _, _, _                  = es.RhythmExtractor2013()(track.track_mono_44)
        beat_loudness, loudness_band_ratio = es.BeatsLoudness(beats = beats)(track.track_mono_44)

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

        track_mono     = track.track_mono_44
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
        tristimulus    = es.Tristimulus()
        chord_detect   = es.ChordsDetection()
        pitch_salience = es.PitchSalience()

        track_mono     = track.track_mono_44
        frame_size     = 1024

        # Apparently I'm supposed to gather these...
        pcp_list       = []
        key_list       = []
        pitch_list     = []
        timbre_list    = []
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
            timbre           = tristimulus(freq, magnitudes)
            key, scale, _, _ = key_extract(pcp)
            pitch            = pitch_salience(spectrum_res)

            pcp_list.append(pcp)
            key_list.append((key, scale))       # TODO : Refactor to a "".join(list)
            pitch_list.append(pitch)            #        for more efficiency...
            timbre_list.append(timbre)
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
        track.features['pitch_salience']     = np.mean(pitch_list, axis=0)
        track.features['tristimulus']        = tuple(np.mean(np.array(timbre_list), axis=0))
        # Btw tristimulus and timbre are interchaneable. But the 'timbre' feature is reserved for models


    @staticmethod
    def get_spectral_centroid_time(track : Track):
        """Computes the spectral centroid across the track using SpectralCentroidTime"""
        window   = es.Windowing(type='hann') 
        spectrum = es.Spectrum()             
        centroid = es.SpectralCentroidTime() 

        frame_size = 1024
        track_mono = track.track_mono_44
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
        track.features['spectral_centroid_mean']    = np.mean(centroids)
        track.features['spectral_centroid_std']     = np.std(centroids)
        track.features['spectral_centroid_max']     = np.max(centroids)

    @staticmethod
    def get_spectral_rolloff(track: Track):
        """Computes the spectral rolloff (frequency below which 85% of energy lies"""
        window      = es.Windowing(type='hann') 
        spectrum    = es.Spectrum()             
        rolloff     = es.RollOff()              

        frame_size  = 1024
        track_mono  = track.track_mono_44
        rolloffs     = []

        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec = spectrum(windowed)
            rolloffs.append(rolloff(spec))
        
        rolloffs = np.array(rolloffs)
        track.features['rolloff_mean'] = np.mean(rolloffs)
        track.features['rolloff_std']  = np.std(rolloffs)
        track.features['rolloff_max']  = np.max(rolloffs)

    @staticmethod
    def get_spectral_contrast(track : Track):
        """Computes Spectral Contrast across several frequency bands"""
        frame_size  = 1024
        window      = es.Windowing(type='hann') 
        spectrum    = es.Spectrum()             
        # Adding the framme_size parameter since it defaults to 2048 and gives an error
        contrast    = es.SpectralContrast(frameSize=frame_size)              

        track_mono  = track.track_mono_44
        contrasts   = []
        valleys     = []

        assert track_mono is not None, "track_mono must not be None"
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
        track.features['spectral_contrast_mean']    = np.mean(contrasts, axis=0)
        track.features['spectral_contrast_std']     = np.std(contrasts, axis=0)
        track.features['spectral_valley_mean']      = np.mean(valleys, axis=0)
        track.features['spectral_valley_std']       = np.std(valleys, axis=0)
          

    @staticmethod
    def get_hfc(track: Track):
        """Compute the High Frequency Content of the track."""
        window   = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        hfc      = es.HFC()

        frame_size = 1024
        track_mono = track.track_mono_44
        hfc_vals   = []

        assert track_mono is not None, "track_mono must not be none"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            hfc_vals.append(hfc(spec))

        hfc_vals = np.array(hfc_vals)
        track.features['hfc_mean'] = np.mean(hfc_vals)
        track.features['hfc_std']  = np.std(hfc_vals)

    @staticmethod
    def get_flux(track: Track):
        """Compute spectral flux over the track."""
        window   = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        flux     = es.Flux()

        frame_size = 1024
        track_mono = track.track_mono_44
        flux_vals  = []

        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            flux_values = flux(spec)
            flux_vals.append(flux_values)


        flux_vals = np.array(flux_vals)
        track.features['flux_mean'] = np.mean(flux_vals)
        track.features['flux_std']  = np.std(flux_vals)

    @staticmethod
    def get_flatness_db(track: Track):
        """Compute spectral flatness in dB."""
        window    = es.Windowing(type='hann')
        spectrum  = es.Spectrum()
        flatness  = es.FlatnessDB()

        frame_size = 1024
        track_mono = track.track_mono_44
        flatness_vals = []

        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i:i+frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            flatness_vals.append(flatness(spec))

        flatness_vals = np.array(flatness_vals)
        track.features['flatness_db_mean'] = np.mean(flatness_vals)
        track.features['flatness_db_std']  = np.std(flatness_vals)

    @staticmethod
    def get_energy_band_ratio(track: Track):
        """Compute energy band ratios for defined frequencies."""


        window     = es.Windowing(type='hann')
        spectrum   = es.Spectrum()

        frame_size = 1024
        track_mono = track.track_mono_44

        band_ratios = []

        # TODO Maybe change 22,050 to 16,000 if there's tracks with lower freq
        freq_bands = [(0, 60), (60, 250),
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


        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)
            
            ratios = [band_algo(spec) for band_algo in energy_band]
            band_ratios.append(ratios)

        ratios_array = np.array(band_ratios)
        print(ratios_array.shape)
        for idx, name in enumerate(band_names):
            track.features[f'energy_ratio_{name}_mean'] = np.nanmean(ratios_array[:, idx])
            track.features[f'energy_ratio_{name}_std']  = np.nanstd(ratios_array[:, idx])


    @staticmethod
    def get_spectral_peaks(track: Track):
        """Extract spectral peaks (frequencies and magnitudes)."""
        window    = es.Windowing(type='blackmanharris92')
        spectrum  = es.Spectrum()
        spec_peaks = es.SpectralPeaks()

        frame_size = 1024
        track_mono = track.track_mono_44
        peak_counts, freqs, mags = [], [], []

        assert track_mono is not None, "track_mono must not be None"
        for i in range(0, len(track_mono), frame_size):
            frame = track_mono[i: i + frame_size]
            if len(frame) < frame_size:
                continue
            windowed = window(frame)
            spec     = spectrum(windowed)

            frequency, magnitude = spec_peaks(spec)
            peak_counts.append(len(frequency))
            if frequency:
                freqs.extend(frequency)
                mags.extend(magnitude)

        if freqs:
            track.features['spectral_peaks_avg_freq']  = np.mean(freqs)
            track.features['spectral_peaks_avg_mag']   = np.mean(mags)
            track.features['spectral_peaks_count']     = np.mean(peak_counts)
        else:
            track.features['spectral_peaks_avg_freq']  = 0
            track.features['spectral_peaks_avg_mag']   = 0
            track.features['spectral_peaks_count']     = 0

