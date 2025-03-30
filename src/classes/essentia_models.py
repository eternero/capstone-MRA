"""Classes and constants for representing Essentia ML Models and Embeddings
NOTE : For this simplicity, this could be considered being turned to a Dataclass.
"""

from typing import Any

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
        algorithm     : str,
        target_index  : int             = 0,
        model_family  : str             = None,
        classifiers   : list[str]       = None,
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
        self.model_family   = model_family

        self.embeddings     = embeddings
        self.classifiers    = classifiers
        self.target_index   = target_index
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
# Approachability and Engagement Models
# ------------------------------------------------------------------------------
approachability_2c              = EssentiaModel(
    graph_filename = 'src/models/approachability_2c-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['not_approachable', 'approachable'],
    target_index   = 1,
    model_family   = "effnet"
)

approachability_3c              = EssentiaModel(
    graph_filename = 'src/models/approachability_3c-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['not_approachable', 'moderately_approachable', 'approachable'],
    model_family   = "effnet"
)

approachability_regression      = EssentiaModel(
    graph_filename = 'src/models/approachability_regression-discogs-effnet-1.pb',
    output         = 'model/Identity',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['approachability'],
    model_family   = "effnet"
)

engagement_2c                    = EssentiaModel(
    graph_filename = 'src/models/engagement_2c-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['not_engaging', 'engaging'],
    target_index   = 1,
    model_family   = "effnet"
)

engagement_3c                   = EssentiaModel(
    graph_filename = 'src/models/engagement_3c-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['not_engaging', 'moderately_engaging', 'engaging'],
    model_family   = "effnet"
)

engagement_regression           = EssentiaModel(
    graph_filename = 'src/models/engagement_regression-discogs-effnet-1.pb',
    output         = 'model/Identity',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['engagement'],
    model_family   = "effnet"
)

# ------------------------------------------------------------------------------
# Effnet Models
# ------------------------------------------------------------------------------
danceability_effnet_model       = EssentiaModel(
    graph_filename = 'src/models/danceability-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['danceable', 'not_danceable'],
    model_family   = "effnet"
)

mood_aggressive_effnet_model    = EssentiaModel(
    graph_filename = 'src/models/mood_aggressive-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['aggressive', 'not_aggressive'],
    model_family   = "effnet"
)

mood_happy_effnet_model         = EssentiaModel(
    graph_filename = 'src/models/mood_happy-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['happy', 'non_happy'],
    model_family   = "effnet"
)

mood_party_effnet_model         = EssentiaModel(
    graph_filename = 'src/models/mood_party-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['non_party', 'party'],    # Metadata has it like this flipped, idk.
    target_index   = 1,                         # `target_index` is a stupid fix for this.
    model_family   = "effnet"
)

mood_relaxed_effnet_model       = EssentiaModel(
    graph_filename = 'src/models/mood_relaxed-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['non_relaxed', 'relaxed'],    # Also flipped man idk.
    target_index   = 1,
    model_family   = "effnet"
)

mood_sad_effnet_model           = EssentiaModel(
    graph_filename = 'src/models/mood_sad-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['non_sad', 'sad'],
    target_index   = 1,
    model_family   = "effnet"
)

mood_acoustic_effnet_model      = EssentiaModel(    # NOTE : TAKES A MELSPECTOGRAM AS EMBEDDING INPUT. FUCK
    graph_filename = 'src/models/mood_acoustic-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['acoustic', 'non_acoustic'],
    model_family   = "effnet"
)

mood_electronic_effnet_model    = EssentiaModel(
    graph_filename = 'src/models/mood_electronic-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['electronic', 'non_electronic'],
    model_family   = "effnet"
)

voice_instrumental_effnet_model = EssentiaModel(
    graph_filename = 'src/models/voice_instrumental-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['instrumental', 'voice'],
    model_family   = "effnet"
)

# NOTE : Could be a good idea to zero-weight this if the instrumental value is v high.
voice_gender_effnet_model       = EssentiaModel(
    graph_filename = 'src/models/gender-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['female', 'male'],
    model_family   = "effnet"
)


tonal_atonal_effnet_model       = EssentiaModel(
    graph_filename = 'src/models/tonal_atonal-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['tonal', 'atonal'],
    model_family   = "effnet"
)

timbre_effnet_model             = EssentiaModel(
    graph_filename = 'src/models/timbre-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['bright', 'dark'],
    model_family   = "effnet"
)

nsynth_timbre_effnet_model      = EssentiaModel(
    graph_filename = 'src/models/nsynth_bright_dark-discogs-effnet-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['bright', 'dark'],
    model_family   = "nsynth_effnet"
)

# ------------------------------------------------------------------------------
# MusiCNN Models
# ------------------------------------------------------------------------------
danceability_musicnn_model = EssentiaModel(
    graph_filename = 'src/models/danceability-msd-musicnn-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['danceable', 'not_danceable'],
    model_family   = "MusiCNN"
)

voice_instrumental_musicnn_model = EssentiaModel(
    graph_filename = 'src/models/voice_instrumental-msd-musicnn-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['instrumental', 'voice'],
    model_family   = "MusiCNN"
)

mood_happy_musicnn_model = EssentiaModel(
    graph_filename = 'src/models/mood_happy-msd-musicnn-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['happy', 'non_happy'],
    model_family   = "MusiCNN"
)

mood_aggressive_musicnn_model = EssentiaModel(
    graph_filename='src/models/mood_aggressive-msd-musicnn-1.pb',
    output         = 'model/Softmax',
    algorithm      = 'TensorflowPredict2D',
    classifiers    = ['aggressive', 'non_aggressive'],
    model_family   = "MusiCNN"
)

# ------------------------------------------------------------------------------
# Create our dictionary of Embeddings and Models
# ------------------------------------------------------------------------------
essentia_models_dict =  {
                        discogs_effnet_emb: [
                                              timbre_effnet_model,
                                              mood_acoustic_effnet_model,
                                              danceability_effnet_model,
                                              voice_instrumental_effnet_model,

                                            ],
                        msd_musicnn_emb:    [
                                              danceability_effnet_model,
                                              voice_instrumental_musicnn_model,
                                              mood_happy_musicnn_model,
                                              mood_aggressive_musicnn_model
                                            ]
                        }
