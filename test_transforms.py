"""
Module: numpy_spectral_analysis.py

Description:
    This module processes a set of audio tracks to extract
    information using two approaches implemented with numpy:
      1. Fourier Transform Analysis: Computes time–frequency representations
         for each track frame and derives features such as spectral centroid, 
         peak frequency, and spectral flux.
      2. Constant-Q Transform Analysis: Computes an approximate constant‐Q
         transform using logarithmically spaced kernels and derives summary
         statistics from the transform coefficients.

    The extracted features from both methods are aggregated into a Pandas DataFrame
    and exported to a CSV file.

Usage:
    - 'track_paths' list with the full paths to audio files.
    - Run the module directly. Audio tracks will be processed, and results saved as CSV.

Dependencies:
    - numpy
    - pandas
    - src.classes.track
    - src.classes.essentia_algos
"""

import numpy as np
import pandas as pd
from src.classes.track import Track, TrackPipeline
from src.classes.essentia_algos import EssentiaAlgo as es


def get_fourier_transform_info(audio, sr, frame_size=2048, hop_size=512):
    """
    Computes Fourier Transform-based spectral features from audio signal.
    
    The audio signal is framed and transformed using FFT.
    For each frame, the function computes:
      - Spectral centroid: a weighted mean frequency.
      - Peak frequency: frequency bin with the maximum magnitude.
      - Spectral flux: sum of squared differences between consecutive frames.
      
    Parameters:
        audio (np.array): 1D array of audio samples.
        sr (int): Sample rate.
        frame_size (int): Number of samples per frame.
        hop_size (int): Step size between consecutive frames.
    
    Returns:
        dict: Aggregated features including:
              'spectral_centroid', 'mean_peak_freq', 'avg_spectral_flux'
    """
    num_frames = 1 + max(0, (len(audio) - frame_size) // hop_size)
    spectral_centroids = []
    peak_freqs = []
    spectral_flux = []
    prev_magnitude = None

    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
        window = np.hanning(frame_size)
        frame_windowed = frame * window

        # Compute FFT and magnitude spectrum
        spectrum = np.fft.rfft(frame_windowed)
        magnitude = np.abs(spectrum)
        frequencies = np.fft.rfftfreq(frame_size, d=1/sr)

        # Spectral Centroid: weighted average of frequencies
        centroid = np.sum(frequencies * magnitude) / (np.sum(magnitude) + 1e-10)
        spectral_centroids.append(centroid)

        # Peak Frequency: frequency with maximum magnitude
        peak_freq = frequencies[np.argmax(magnitude)]
        peak_freqs.append(peak_freq)

        # Spectral Flux: squared difference between successive magnitudes
        if prev_magnitude is not None:
            flux = np.sum((magnitude - prev_magnitude) ** 2)
            spectral_flux.append(flux)
        prev_magnitude = magnitude

    features = {
        'spectral_centroid': np.mean(spectral_centroids),
        'mean_peak_freq': np.mean(peak_freqs),
        'avg_spectral_flux': np.mean(spectral_flux) if spectral_flux else 0.0
    }
    return features


def constant_q_transform(audio, sr, fmin=55.0, bins_per_octave=12, n_octaves=5, Q=None):
    """
    Computes a simplified Constant-Q Transform for audio signal.
    
    This implementation generates logarithmically spaced frequency bins between
    fmin and fmin*(2^(n_octaves)). For each bin, a kernel is created whose length 
    is proportional to Q * sr/f0. The inner product of the kernel with an appropriate 
    segment of the audio yields one Constant-Q coefficient.
    
    Parameters:
        audio (np.array): 1D array of audio samples.
        sr (int): Sample rate.
        fmin (float): Minimum frequency (in Hz) for analysis.
        bins_per_octave (int): Number of bins in one octave.
        n_octaves (int): Total number of octaves to analyze.
        Q (float, optional): Quality factor; if None, computed as 1/(2^(1/bins_per_octave)-1).
    
    Returns:
        np.array: Array of complex Constant-Q transform coefficients.
    """
    if Q is None:
        Q = 1 / (2**(1/bins_per_octave) - 1)
    n_bins = bins_per_octave * n_octaves
    cqt_coeffs = np.zeros(n_bins, dtype=np.complex64)

    for k in range(n_bins):
        # Compute center frequency for bin k
        f0 = fmin * (2 ** (k / bins_per_octave))
        # Window length for this bin
        L = int(np.ceil(Q * sr / f0))
        if L > len(audio):
            padded_audio = np.pad(audio, (0, L - len(audio)), 'constant')
        else:
            start = (len(audio) - L) // 2
            padded_audio = audio[start:start + L]
        # Create kernel: Hann window modulated with a complex exponential
        window = np.hanning(L)
        n = np.arange(L)
        kernel = window * np.exp(-2j * np.pi * f0 * n / sr)
        cqt_coeffs[k] = np.dot(padded_audio, kernel)
    return cqt_coeffs

def get_constant_q_info(audio, sr, **kwargs):
    """
    Computes Constant-Q based features from an audio signal.
    
    It uses the constant_q_transform function to obtain transform coefficients and 
    then derives summary statistics.
    
    Parameters:
        audio (np.array): 1D audio sample array.
        sr (int): Sample rate.
        kwargs: Additional parameters for constant_q_transform.
    
    Returns:
        dict: A dictionary with features:
              - 'cqt_magnitude_mean'
              - 'cqt_magnitude_std'
              - 'cqt_max_coeff'
    """
    cqt_coeffs = constant_q_transform(audio, sr, **kwargs)
    magnitudes = np.abs(cqt_coeffs)
    features = {
        'cqt_magnitude_mean': np.mean(magnitudes),
        'cqt_magnitude_std': np.std(magnitudes),
        'cqt_max_coeff': np.max(magnitudes)
    }
    return features

def process_track(track_path):
    """
    Processes a single audio track using NumPy-based spectral analysis.

    Steps:
      1. Initializes the Track object and converts it to mono at 44100 Hz.
      2. Applies Fourier Transform-based analysis.
      3. Applies Constant-Q Transform analysis.
      4. Aggregates and rounds the extracted features.

    Parameters:
        track_path (str): Full path to the audio file.
    """
    track = Track(track_path)
    # Get a mono version of the track at 44100 Hz sample rate.
    track.track_mono_44 = track.get_track_mono(44100)
    audio = track.track_mono_44
    sr = 44100

    # Extract spectral features using NumPy-based approaches.
    ft_features = get_fourier_transform_info(audio, sr)
    cq_features = get_constant_q_info(audio, sr, fmin=55.0, bins_per_octave=12, n_octaves=5)

    # Merge the features into the track's feature dictionary.
    track.features.update(ft_features)
    track.features.update(cq_features)

    return track


if __name__ == "__main__":
    track_paths = [
        "/Users/edgaeldejesus/Downloads/Capstone_Playlist/03 Wet Dreamz.flac",
        "/Users/edgaeldejesus/Downloads/Capstone_Playlist/04 - Natural.flac",
        "/Users/edgaeldejesus/Downloads/Capstone_Playlist/06 - Post Malone - White Iverson (Explicit).flac",
    ]
    
    all_results = {}
    
    # Process each track using the custom NumPy-based processing function.
    for track_path in track_paths:
        processed_track = process_track(track_path)
        track_name = track_path.split("/")[-1].replace(".flac", "")
        # Round numerical features for a cleaner output.
        all_results[track_name] = {
            feature: round(float(value), 4) if isinstance(value, (float, int, np.floating)) else value
            for feature, value in processed_track.features.items()
        }
    
    # Aggregate results into a Pandas DataFrame and export to CSV.
    df = pd.DataFrame.from_dict(all_results, orient='index')
    print(df)
    df.to_csv("numpy_spectral_features.csv", index=True)
