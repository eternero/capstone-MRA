"""
...
"""

from src.classes.track import TrackPipline

if __name__ == '__main__':
    AUDIO_PATH = "src/audio/dataset_flac"

    track_pipeline = TrackPipline(AUDIO_PATH)

    print("Loading tracks in parallel...")
    track_pipeline.load_tracks(num_processes=8)  # Adjust number of processes as necessary. 
                                                 # you could also just leave it at default.

    print(f"Loaded {len(track_pipeline.track_list)} tracks.")
