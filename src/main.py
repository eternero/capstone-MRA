"""..."""
import pandas as pd
from src.classes.track import TrackPipeline
from src.classes.essentia_containers import essentia_task_list

if __name__ == '__main__':
    AUDIO_PATH_LIST = ["src/audio/dataset_flac_1", "src/audio/dataset_flac_2",
                       "src/audio/dataset_flac_3", "src/audio/dataset_flac_4",
                       "src/audio/dataset_flac_5", "src/audio/dataset_flac_6"
                      ]

    track_df_list = []
    for ix, audio_path in enumerate(AUDIO_PATH_LIST):
        track_pipeline = TrackPipeline(audio_path)
        track_list     = track_pipeline.run_pipeline(essentia_task_list = essentia_task_list,
                                                     additional_tasks   = None,
                                                     pooling            = True,
                                                     segment_position   = 1,
                                                     segment_size       = 5)

        track_df   = track_pipeline.get_track_dataframe()
        track_df_list.append(track_df)


    all_track_df = pd.concat(track_df_list)
    all_track_df = all_track_df.drop_duplicates(subset=['clean_album','clean_title'])
    all_track_df.to_csv('dataset_pooled_5s_mod.csv', index=False)


    df = pd.read_csv('datasets/dataset_flac_10s_pooled_mod2.csv')
    df = df.drop_duplicates(subset=['clean_album','clean_title'])
    df.to_csv('dataset_pooled_10s_mod3.csv', index=False)