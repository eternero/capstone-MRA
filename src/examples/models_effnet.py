"""
...
"""

from pprint import pprint
from essentia.standard import (MonoLoader, 
                               TensorflowPredictEffnetDiscogs,
                               TensorflowPredict2D)
from src.classes.essentia_models import (discogs_effnet_emb,
                                         timbre_effnet_model,
                                         danceability_effnet_model)

track_path = "src/audio/dataset_flac/01 Dead.flac"

audio = MonoLoader(filename=track_path, sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="src/embeddings/discogs-effnet-bs64-1.pb",
                                                 output="PartitionedCall:1")
embeddings = embedding_model(audio)

print("LOADED EMBEDDINGS")


model = TensorflowPredict2D(graphFilename="src/models/timbre-discogs-effnet-1.pb", output="model/Softmax")(embeddings)
print("LOADED INFERENCE MODEL")


# predictions = model(embeddings)

# pprint(predictions)
