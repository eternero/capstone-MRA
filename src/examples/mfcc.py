import time
import torch
import torchaudio
import numpy as np
from essentia.standard import MonoLoader
from src.utils.parallel import load_essentia_algorithm

def get_track_mfcc(track_mono  : np.ndarray,
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
    mfcc_array = np.vstack(mfcc_list)


    features = {
        'mfcc_mean' : np.mean(mfcc_array, axis=0),
        'mfcc_std'  : np.std(mfcc_array, axis=0),

        'band_mean' : np.mean(bands_array, axis=0),
        'band_std'  : np.std(bands_array, axis=0),
    }

    return features


TRACK_PATH = '/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/src/audio/testing_dataset_flac/01 - AAA Powerline.flac'

start_1 = time.time()
track_mono     = MonoLoader(filename = TRACK_PATH, sampleRate = 44100, resampleQuality = 0)()
get_track_mfcc(track_mono=track_mono)
print(f"Executed in {time.time() - start_1}s")

start_2  = time.time()
sr       = 44100
segments = [(0, 20), (70,80), (130, 140), (190, 200)] # These are start_sec, end_sec

for segment in segments:
    start_sec, end_sec = segment
    duration_sec = end_sec - start_sec # Should be 10 seconds

    # Calculate frame offset and number of frames correctly
    frame_offset = int(start_sec * sr)
    num_frames_to_load = int(duration_sec * sr) # Load exactly 10 seconds worth

    try:
        # Load the segment using torchaudio
        track_seg_tensor, loaded_sr = torchaudio.load(
            TRACK_PATH,
            frame_offset=frame_offset,
            num_frames=num_frames_to_load
        )

        # Ensure sample rate matches expectation (torchaudio doesn't resample on load)
        if loaded_sr != sr:
            print(f"Warning: Loaded SR {loaded_sr} differs from expected {sr}. Resampling.")
            resampler = torchaudio.transforms.Resample(loaded_sr, sr)
            track_seg_tensor = resampler(track_seg_tensor)

        # Convert to mono if necessary
        if track_seg_tensor.shape[0] > 1:
            track_seg_tensor = torch.mean(track_seg_tensor, dim=0, keepdim=True)

        # --- PROBLEM AREA ---
        # get_track_mfcc expects a NumPy array, but track_seg_tensor is a PyTorch Tensor.
        # Convert the tensor to a NumPy array.
        # Squeeze removes the channel dimension (if mono), detach prevents gradient tracking.
        track_seg_numpy = track_seg_tensor.squeeze().detach().numpy()

        # Ensure it's float32, as often expected by Essentia/audio libraries
        track_seg_numpy = track_seg_numpy.astype(np.float32)

        # Now pass the NumPy array to the function
        get_track_mfcc(track_mono=track_seg_numpy)

    except Exception as e:
        print(f"Error processing segment {start_sec}-{end_sec}s: {e}")
        # Decide how to handle errors, e.g., continue to next segment


print(f"Executed in {time.time() - start_2}s")
