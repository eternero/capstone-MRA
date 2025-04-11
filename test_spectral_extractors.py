import numpy as np
import pandas as pd  # Add this for nice table formatting
from src.classes.track import Track, TrackPipeline
from src.classes.essentia_algos import EssentiaAlgo
from src.classes.essentia_containers import EssentiaAlgorithmTask

# Pathnames of freaking songs
track_paths = [
    "/Users/jorge/soulseek-music/02. Mr. Brightside.flac",
    "/Users/jorge/soulseek-music/02-arctic_monkeys-r_u_mine.flac",
    "/Users/jorge/soulseek-music/01. Dog Days Are Over.flac",
    "/Users/jorge/soulseek-music/Billie Eilish_WHEN WE ALL FALL ASLEEP, WHERE DO WE GO!_10_bury a friend.flac",
]

audio_path = "/Users/jorge/soulseek-music"

# Algorithms to test :D
feature_extractors = [
    EssentiaAlgo.get_spectral_centroid_time,
    EssentiaAlgo.get_spectral_rolloff,
    EssentiaAlgo.get_spectral_contrast,
    EssentiaAlgo.get_hfc,
    EssentiaAlgo.get_flux,
    EssentiaAlgo.get_flatness_db,
    EssentiaAlgo.get_energy_band_ratio,
    EssentiaAlgo.get_spectral_peaks
]

essentia_algoritms_task = EssentiaAlgorithmTask(feature_extractors)

all_results = {}


def process_track(track_path):
    track = Track(track_path)
    track.track_mono_44 = track.get_track_mono(44100)

    track_name = track_path.split("/")[-1].replace(".flac", "")

    for extractor in feature_extractors:
        extractor(track)

    all_results[track_name] = {}
    for feature, value in track.features.items():
        if isinstance(value, (float, int, np.floating)):
            all_results[track_name][feature] = round(float(value), 4)
        else:
            all_results[track_name][feature] = value


if __name__ == "__main__":

    # for track_path in track_paths:
    #     process_track(track_path)

    # df = pd.DataFrame.from_dict(all_results, orient='index')
    # print(df)

    # df.to_csv('track_features_summary.csv')


    track_pipeline = TrackPipeline(audio_path)
    track_pipeline.run_pipeline(essentia_task_list=[essentia_algoritms_task])
    track_df = track_pipeline.get_track_dataframe()
    track_df.to_csv("track_features_summary_pl.csv", index=True)
