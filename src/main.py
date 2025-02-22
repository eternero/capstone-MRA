"""
...
"""
import os
import time
from pprint import pprint
from src.classes.track import TrackPipeline
from src.classes.essentia_models import (discogs_effnet_emb,
                                         timbre_effnet_model,
                                         acoustic_effnet_model,
                                         danceability_effnet_model,
                                         voice_instrumental_effnet_model
)

if __name__ == '__main__':
    print(os.cpu_count())
    AUDIO_PATH = "src/audio/dataset_flac"
    track_pipeline = TrackPipeline(AUDIO_PATH)


    # print("Loading tracks in parallel...")
    # track_pipeline......(num_processes=15)  # Adjust number of processes as necessary.
    #                                              # you could also just leave it at default.


    # n_tracks | ppe | pipeline (tpe)   | ppe mp3
    # 20       | 21  |                  |
    # 100      | 94  | 227 haha         | 89

    essentia_models_dict =  {
                            discogs_effnet_emb: [
                                                timbre_effnet_model,
                                                acoustic_effnet_model,
                                                danceability_effnet_model,
                                                voice_instrumental_effnet_model
                                                ]
                            }
    track_pipeline.run_pipeline(essentia_models_dict)
    track_df = track_pipeline.get_track_dataframe()
    track_df.to_csv('track_df.csv', index=False)



    sample_track = track_pipeline.track_list[0].features
    pprint(sample_track)


    # After running this, tracks should now have features and metadata
    # track_pipeline.run_pipeline(essentia_models_dict)
    # for track in track_pipeline.track_list:
    #     pprint(track.metadata)

    # TODO : This was surprisingly slow and some models seemed to be loaded twice, which should
    #        never happen. Analyze and figure out why this is happening. A good place to start
    #        is by running a couple of tracks manually first. Create an `/examples` dir for this.
    #
    #        Seems that the double loading is normal...




# PARALLEL
# MODEL : src/models/timbre-discogs-effnet-1.pb - took 51.669508934020996.

# LINEAR
# MODEL : src/models/timbre-discogs-effnet-1.pb - took 49.821990966796875.





"""
Grimes - Butterfly 


(effnet)
bright / dark               acoustic / non acoustic     danceability                instrumental / voice [?]
[0.451644   0.54835606]	    [0.0949063 0.9050941]       [0.83635575 0.16364348]	    [0.8217883  0.17821142]


(spotify)
- Danceability     : 0.697          [ Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity]
- Speechiness      : 0.0296         [ Speechiness detects the presence of spoken words in a track. ]
- Instrumentalness : 0.0622         [ Likeliness that the track is an instrumental. No words that is. ]
- Acousticness     : 0.00131        [ A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic ]


Now that I see the tags in more detail for Spotify, they seem to hard-focus on stuff, rather than indicate the presence of elements.
Speechiness, Instrumentalness and Acousticness are all hard-focused tags for Spoken Word, Instrumental and Acoustic respectively.

Even if we disregard these (which I don't think we should), we can still acquire this information through descriptors, whether it
be from RYM or using some of the Essentia Models. It is pretty interesting though, the Audio Features from the Spotify API are not
nearly as useful as I had initially thought they'd be, or at least these. Other elements like energy, tempo, key, loudness, mode
could still be useful to us.


Burial - Ethched Headplate

(effnet)
- timbre       : 0.55 dark
- acoustic     : 0.878 non-acoustic (True, the track is completely digital, it's fucking electronic music)
- danceability : 0.867 (not good, but we could make our own danceability equation if needed!)
- instrumental : 0.826 instrumental, 0.174 voice (very right and super close to spotify [0.172])


(spotify)
- Danceability     : 0.421          
- Speechiness      : 0.172         
- Instrumentalness : 0.726         
- Acousticness     : 0.279        


"""