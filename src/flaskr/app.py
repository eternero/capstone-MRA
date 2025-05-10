"""TODO Add File Docstring"""

import os
import time
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

from src.classes.track import TrackPipeline
from src.classes.essentia_containers import essentia_task_list
from src.classes.distance import (DistPipeline, DistMethods,
                                  z_score_normalization, num_features, dim_features)

# -------------------------------------------------------------------------------------------------
#  Define Flask Environment / Config Variables / Constants
# -------------------------------------------------------------------------------------------------
DATASET_PATH  = 'datasets/pooled_dataset.csv'
UPLOAD_FOLDER = 'src/flaskr/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'flac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 40 * 1000 * 1000     # 40MB Max File Upload Size


def allowed_file(filename: str) -> bool:
    """Basic method for file type validation."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def index():
    """Method to render our home page."""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # 1. Validate upload field
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 2. Validate extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # 3. Secure and temporarily save to disk
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)


    # 4. Process the Input Track with the `TrackPipeline`
    track_pipeline = TrackPipeline(base_path = filepath)
    track_pipeline.run_pipeline(essentia_task_list = essentia_task_list,
                                additional_tasks   = None, pooling = True)
    input_track_df = track_pipeline.get_track_dataframe()

    # 5. Get the top recommendations for the track...
    dist_pipeline  = DistPipeline(input_track_df       = input_track_df,
                                  track_dataset_path   = DATASET_PATH,
                                  numerical_dist       = DistMethods.cosine_numerical,
                                  dimensional_dist     = DistMethods.cosine_dimensional,
                                  numerical_features   = num_features,
                                  dimensional_features = dim_features,
                                  normalize_numerical  = z_score_normalization,
                                  pooling              = False
                                  )


    start = time.time()
    top_recs       = dist_pipeline.run_pipeline_parallel(top_n = 10)
    print(f"Executed in {time.time() - start}")

    # 6. Remove file once it has been used.
    os.remove(filepath)

    # 6) Send JSON back
    return jsonify(top_recs)

if __name__ == '__main__':
    app.run(debug=True)
