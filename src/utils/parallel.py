"""
File which gathers methods to be used for parallelism.
"""
import os
from functools import lru_cache
from typing import Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
import torchaudio
import numpy as np
import essentia.standard as es
from src.external.harmof0 import harmof0



def torch_load(track_path : str, seg_start : int) -> tuple[np.ndarray]:
    """Wrapper for `torchaudio.load()` which will load a track as a 44.1kHz and 16kHz Numpy Mono Array"""

    # We assume a sample rate of 44.1kHz and segment_size = 10s
    seg_size               = 10
    track_seg_tensor_44, _ = torchaudio.load(track_path,
                                          frame_offset = seg_start * 44100,
                                          num_frames   = seg_size  * 44100)

    # Create a resampled track from 44.1kHz to 16kHz
    resampler              = torchaudio.transforms.Resample(44100, 16000)
    track_seg_tensor_16    = resampler(track_seg_tensor_44)


    # Convert both to a mono (1-Dim) np.ndarray
    torch_mono_44          = torch.mean(track_seg_tensor_44, dim=0, keepdim=True)
    torch_mono_16          = torch.mean(track_seg_tensor_16, dim=0, keepdim=True)

    np_mono_44             = torch_mono_44.squeeze().detach().numpy().astype(np.float32)
    np_mono_16             = torch_mono_16.squeeze().detach().numpy().astype(np.float32)

    return np_mono_44, np_mono_16


@lru_cache(maxsize=32)
def load_essentia_algorithm(algorithm_name : str, *args, **kwargs):
    """Allows for Essentia Algoroithms to be loaded and cached within processes"""

    algorithm = getattr(es, algorithm_name)
    return      algorithm(*args, **kwargs)


@lru_cache(maxsize=32)
def load_essentia_model(algorithm_name : str, graph_filename : str, output_name : str):
    """Allows for Essentia Models to be loaded and cached within processes."""
    model_callable = getattr(es, algorithm_name)
    model_tf       = model_callable(graphFilename = graph_filename,
                                    output        = output_name)
    return model_tf

@lru_cache(maxsize=1)
def get_pitch_tracker(device='mps'):
    """Allows caching for the harmof0 PitchTracker object."""
    return harmof0.PitchTracker(device=device)

def run_in_parallel(func         : Callable,
                    item_list    : list[Any], *args,
                    num_workers  : int = os.cpu_count(),
                    executor_type: str = "process",
                    **kwargs)   -> list[Any]:
    """
    Runs the provided function in parallel for each item in item_list. This is essentially
    a wrapper of the concurrent.futures XPoolExecutor methods adapted to our use case.

    Args:
        func          : The method to be run in parallel.
        item_list     : Will either be a list of all track filenames, or a list of all tracks.
        num_workers   : Number of workers to use for our executor.
        executor_type : The executor to be used, either ThreadPoolExec or ProcessPoolExec.
    """

    # First, determine which Executor will be used.
    if executor_type.lower() == "process":
        Executor = ProcessPoolExecutor
    else:
        Executor = ThreadPoolExecutor

    results = []    # List to collect all results later on.
    with Executor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(func, item, *args, **kwargs)
            for item in item_list
        ]

        # Wait for all tasks to complete and handle exceptions.
        for future in as_completed(futures):
            try:
                result = future.result()    # This is done to handle
                results.append(result)      # all results asynchronously
                                            # and individually.
            except Exception as e:
                print(f"Error processing a track: {e}")

    return results
