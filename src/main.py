"""..."""
from src.classes.track import TrackPipeline
from src.classes.essentia_containers import essentia_task_list

if __name__ == '__main__':
    AUDIO_PATH_LIST = ["src/audio/dataset_flac_1", "src/audio/dataset_flac_2",
                       "src/audio/dataset_flac_3", "src/audio/dataset_flac_4",
                       "src/audio/dataset_flac_5", "src/audio/dataset_flac_6"
                      ]

    for ix, audio_path in enumerate(AUDIO_PATH_LIST):
        track_pipeline = TrackPipeline(audio_path)
        track_list     = track_pipeline.run_pipeline(essentia_task_list = essentia_task_list,
                                                     additional_tasks   = None,
                                                     pooling            = True)

        track_df   = track_pipeline.get_track_dataframe()
        track_df.to_csv(f'dataset_flac_{ix}_pooled.csv', index=False)
