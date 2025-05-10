"""..."""
import os
import re
import json
import time
import glob
import shutil
import subprocess
import unicodedata
from functools import wraps
from functools import lru_cache
from dataclasses import replace
from collections import defaultdict
from typing import Any, Callable, List, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
import spotipy
import torchaudio
import numpy as np
import pandas as pd
import essentia.standard as es
from spotipy.oauth2 import SpotifyOAuth

if TYPE_CHECKING:
    from src.classes.track import Track

# -------------------------------------------------------------------------------------------------
# Stupid small helpers.
# -------------------------------------------------------------------------------------------------

def load_json(val):
    """Helper: safely load JSON strings, returning an empty list on error."""
    try:
        return json.loads(val)
    except Exception:
        return []


def is_missing(field):
    """Helper for `analyze_progress`, checks if a field is missing."""
    if not isinstance(field, list) or len(field) == 0:
        return True

    if all(isinstance(x, str) and x.strip() == "" for x in field):
        return True

    return False

# -------------------------------------------------------------------------------------------------
# Metadata Processing Utils.
# -------------------------------------------------------------------------------------------------

def process_name(name: str) -> str:
    """Process album or artist names for consistency while preserving foreign characters."""
    # 0. Lowercase the name.
    name = name.lower()

    # 0.5 Super specific edge case :)
    name = name.replace('℮', 'e')

    # 1. Normalize the string using NFKC (this still composes characters but doesn't force ASCII)
    name = unicodedata.normalize('NFKC', name)

    # 1.5 Second round of normalization.
    decomposed_name = unicodedata.normalize('NFD', name)
    filtered_name   = []
    for char in decomposed_name:
        if unicodedata.combining(char):
            continue
        filtered_name.append(char)
    name = unicodedata.normalize('NFC', "".join(filtered_name))

    # 2. Replace dollar signs with underscore "_"
    name = name.replace('$', '_')

    # 3. Remove anything inside parentheses or square brackets (including the brackets)
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)

    # 4. Replace a period between two word characters with an underscore
    name = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '_', name)

    # 4.5. Replace any remaining periods with a space
    name = re.sub(r'\.', ' ', name)

    # 5. Replace '&' with 'and'
    name = name.replace('&', 'and')

    # 6. Remove all symbols except word characters, whitespace, hyphens, and en-dashes
    name = re.sub(r"[^\w\s\-–]", "", name)

    # 7. Collapse whitespace around hyphens (ensuring no extra spaces around them)
    name = re.sub(r'\s*([-–])\s*', r'\1', name)

    # 8. Collapse multiple spaces into one and trim leading/trailing whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # 9. Replace all remaining spaces with hyphens
    name = name.replace(' ', '-')

    return name

# -------------------------------------------------------------------------------------------------
# Dataframe Alterations
# -------------------------------------------------------------------------------------------------

def concat_dfs(csv_paths : list[str]) -> pd.DataFrame:
    """Stupid wrapper for pandas concat"""
    dataframes  = []
    for csv in csv_paths:
        curr_df = pd.read_csv(csv)
        dataframes.append(curr_df)

    return pd.concat(dataframes)


# -------------------------------------------------------------------------------------------------
# File Coversion (FLAC/MP3,M4A) and Directory Modifications
# -------------------------------------------------------------------------------------------------

def convert_files(source_dir : str, target_dir : str):
    """
    Method to convert out FLAC files into MP3s from one directory to another.
    """
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all .flac files in the source directory
    flac_files = glob.glob(os.path.join(source_dir, "*.flac"))
    for flac_file in flac_files:
        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(flac_file))[0]

        # Define the output filename (MP3)
        mp3_file = os.path.join(target_dir, base_name + ".mp3")

        # Option 1: Use constant bit rate (CBR) at 320 kbps for highest quality
        command = [
            "ffmpeg",
            "-i", flac_file,
            "-b:a", "320k",
            mp3_file
        ]


        print(f"Converting {flac_file} to {mp3_file}...")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {flac_file}: {e}")

    print("Conversion complete!")


def flatten_directory(root_dir : str):
    """Flattens a directory so that all files are stored on the top level.
    Subdirectories will be removed so be careful when running!

    Args:
        root_dir (str): The root directory to be flattened.
    """
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for file in filenames:
            src_path = os.path.join(dirpath, file)
            dest_path = os.path.join(root_dir, file)

            # Handle duplicates
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(root_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dest_path)

    # Clean up empty directories
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)

    print("Flattening complete!")


def convert_m4a_to_flac(input_dir : str, output_dir : str):
    """Copies a directory of M4As and creates a new directory of FLACs.

    Args:
        input_dir  : The original directory which contains the M4A files.
        output_dir : The path to which you want these files to be converted and saved.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .m4a files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".m4a"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".flac")

            # Run ffmpeg to convert the file
            subprocess.run(["ffmpeg", "-i", input_path, "-c:a", "flac", output_path], check=True)

    print("Conversion complete!")


# -------------------------------------------------------------------------------------------------
# Loaders and other utils for the Track Pipeline
# -------------------------------------------------------------------------------------------------

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


def pool_segments(track_list: List["Track"]) -> List["Track"]:
    """
    Given a list of Track segments (where many Track objects share the same track_path),
    return a new list of Track objects—one per unique path—with their features pooled
    (mean) across all segments.
    """
    # 1) Group segments by original file path
    groups: dict[str, List["Track"]] = defaultdict(list)
    for seg in track_list:
        groups[seg.track_path].append(seg)

    pooled: List[Track] = []

    # 2) For each group, average its features
    for _, segs in groups.items():
        # Take the first segment as a template for metadata, uri, etc.
        template = segs[0]
        # Determine which feature keys exist
        feat_keys = template.features.keys()

        new_feats: dict[str, Any] = {}

        for key in feat_keys:

            vals = [s.features[key] for s in segs]

            # Vector feature (list or ndarray)?
            if isinstance(vals[0], (list, tuple, np.ndarray)):
                arr = np.stack([np.array(v) for v in vals], axis=0)
                # mean across axis=0 → same shape as one segment
                new_feats[key] = arr.mean(axis=0).tolist()
            else:
                # scalar numerical
                new_feats[key] = float(np.mean(vals))

        # 3) Build a new Track with pooled features (drop segment-specific metadata)
        metadata = {**template.metadata}
        # remove segment_num / segment_start if present
        for seg_key in ("segment_num","segment_start"):
            metadata.pop(seg_key, None)

        pooled_track = replace(
            template,
            features=new_feats,
            metadata=metadata,
        )
        pooled.append(pooled_track)

    return pooled

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


# -------------------------------------------------------------------------------------------------
# Spotify Playlist Creation using our API Creds
# -------------------------------------------------------------------------------------------------

def create_playlist_with_tracks(client_id : str, client_secret : str,
                                redirect_uri : str, playlist_name : str, uri_list : list[str]):
    """Creates a new Spotify Playlist given a list of Track URIs.

    Args:
        client_id     : The Client ID generated after registering your application.
        client_secret : The Client SECRET generated after registering your application.
        redirect_uri  : The callback URI(s) which have been created in your Spotify API Dashboard.
        playlist_name : The name of the playlist.
        uri_list      : The list of Spotify Track URIs. These represent the tracks to be added.
    """

    # Set up authentication
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope='playlist-modify-public'))

    # Retrieve current user's profile information
    user_id = sp.me()['id']

    # Create a new playlist
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
    playlist_id = playlist['id']

    # Add tracks to the playlist in batches of 100 (Spotify's limit per request)
    for i in range(0, len(uri_list), 100):
        sp.playlist_add_items(playlist_id, uri_list[i:i + 100])

    print(f'Playlist "{playlist_name}" created successfully with {len(uri_list)} tracks.')


# -------------------------------------------------------------------------------------------------
#  Decorators
# -------------------------------------------------------------------------------------------------

def timing_decorator(func):
    """
    Simple decorator to measure and print the execution time of a method in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        endt_time   = time.time()
        elapse_time = endt_time - start_time

        print(f"{func.__name__} took {elapse_time:4f} seconds to complete")

        return result
    return wrapper