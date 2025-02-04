"""
...
"""

from pprint import pprint
from src.classes.track import TrackPipline
from src.classes.essentia_models import essentia_models_dict

if __name__ == '__main__':
    AUDIO_PATH = "src/audio/dataset_flac"

    track_pipeline = TrackPipline(AUDIO_PATH)

    print("Loading tracks in parallel...")
    track_pipeline.load_tracks(num_processes=15)  # Adjust number of processes as necessary.
                                                 # you could also just leave it at default.

    print(f"Loaded {len(track_pipeline.track_list)} tracks.")


    print("Reducing number of tracks to 20")
    track_pipeline.track_list = track_pipeline.track_list[:20]

    # After running this, tracks should now have features and metadata
    track_pipeline.run_pipeline(essentia_models_dict)
    for track in track_pipeline.track_list:
        pprint(track.metadata)

    # TODO : This was surprisingly slow and some models seemed to be loaded twice, which should
    #        never happen. Analyze and figure out why this is happening. A good place to start
    #        is by running a couple of tracks manually first. Create an `/examples` dir for this.
