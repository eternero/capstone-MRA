"""Classes and constants for representing Essentia ML Models and Embeddings."""

from typing import Any
from essentia.standard import (
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
    TensorflowPredictEffnetDiscogs,
)



class EssentiaModel:
    """Represents either an Essentia Model or an Embedding.

    This class provides a way to cleanly interact with Essentia models and embeddings. Since models
    and embeddings share certain parameters (e.g. `graph_filename`, `algorithm`, `output`), this
    class can represent both. It also allows a model to reference its corresponding embedding
    object.

    Attributes:
        graph_filename: The file path to the model or embedding graph (.pb file).
        output        : The name of the output node for the graph.
        algorithm     : The TensorFlow algorithm (Essentia wrapper) for this model or embedding.
        embeddings    : The EssentiaModel that references an embedding used by a model. Only used
                        if the current object is an ML Model.
    """

    def __init__(
        self,
        graph_filename: str,
        output        : str,
        algorithm     : Any,
        embeddings    : 'EssentiaModel' = None,
    ) -> None:
        """Initializes an EssentiaModel instance.

        Args:
            graph_filename: The file path to the model or embedding graph (.pb file).
            output        : The name of the output node for the graph.
            algorithm     : The TensorFlow algorithm (Essentia wrapper) for this model or embedding.
            embeddings    : The EssentiaModel that references an embedding used by a model. Only
                            used if the current object is an ML Model.
        """
        self.graph_filename = graph_filename
        self.output         = output
        self.algorithm      = algorithm
        self.embeddings     = embeddings
        self.model          = None

    def get_model(self) -> Any: # Not to be used atm.
        """Returns the instantiated model or embedding algorithm.

        This method calls the Essentia TensorFlow wrapper (e.g., TensorflowPredict2D) and returns
        its instance.

        Returns:
            The instantiated Essentia TensorFlow model/embedding.
        """
        if not self.model: # The addition of this should allow for caching of inference models.
            self.model = self.algorithm(
                graphFilename=self.graph_filename,
                output=self.output,
                )

        return self.model

    def get_algorithm(self) -> str:
        """Returns the algorithm."""
        return self.algorithm

    def get_graph_filename(self) -> str:
        """Returns the graph filename."""
        return self.graph_filename

    def get_output(self) -> str:
        """Returns the output node name."""
        return self.output

    def get_embeddings(self) -> 'EssentiaModel':
        """Returns the embeddings EssentiaModel if it exists."""
        return self.embeddings


# ------------------------------------------------------------------------------
# Discog Embeddings
# ------------------------------------------------------------------------------
discogs_effnet_emb = EssentiaModel(
    graph_filename='src/embeddings/discogs-effnet-bs64-1.pb',
    output='PartitionedCall:1',
    algorithm='TensorflowPredictEffnetDiscogs',
)

# ------------------------------------------------------------------------------
# MusiCNN Embeddings
# ------------------------------------------------------------------------------
msd_musicnn_emb = EssentiaModel(
    graph_filename='src/embeddings/msd-musicnn-1.pb',
    output='model/dense/BiasAdd',
    algorithm='TensorflowPredictMusiCNN',
)

# ------------------------------------------------------------------------------
# Effnet Models
# ------------------------------------------------------------------------------
timbre_effnet_model             = EssentiaModel(
    graph_filename='src/models/timbre-discogs-effnet-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

danceability_effnet_model      = EssentiaModel(
    graph_filename='src/models/danceability-discogs-effnet-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

acoustic_effnet_model           = EssentiaModel(
    graph_filename='src/models/mood_acoustic-discogs-effnet-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

voice_instrumental_effnet_model = EssentiaModel(
    graph_filename='src/models/voice_instrumental-discogs-effnet-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

# ------------------------------------------------------------------------------
# MusiCNN Models
# ------------------------------------------------------------------------------
danceability_musicnn_model = EssentiaModel(
    graph_filename='src/models/danceability-msd-musicnn-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

voice_instrumental_musicnn_model = EssentiaModel(
    graph_filename='src/models/voice_instrumental-msd-musicnn-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

mood_happy_musicnn_model = EssentiaModel(
    graph_filename='src/models/mood_happy-msd-musicnn-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

mood_aggressive_musicnn_model = EssentiaModel(
    graph_filename='src/models/mood_aggressive-msd-musicnn-1.pb',
    output='model/Softmax',
    algorithm='TensorflowPredict2D',
)

# ------------------------------------------------------------------------------
# Create our dictionary of Embeddings and Models
# ------------------------------------------------------------------------------
essentia_models_dict =  {
                        discogs_effnet_emb: [
                                              timbre_effnet_model,
                                              danceability_effnet_model
                                            ],
                        msd_musicnn_emb:    [
                                              danceability_effnet_model,
                                              voice_instrumental_musicnn_model,
                                              mood_happy_musicnn_model,
                                              mood_aggressive_musicnn_model
                                            ]
                        }
