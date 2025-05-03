"""TODO Add File Docstring"""

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

UPLOAD_FOLDER = 'src/flaskr/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'flac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 40 * 1000 * 1000     # 40MB Max File Upload Size


# helper to ensure we only accept the right extensions
def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route('/')
def index():
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
    filename = secure_filename(file.filename)                         # avoid path tricks [oai_citation:2‡Flask Documentation](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/?utm_source=chatgpt.com)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)


    # 4. Run all the pipelines nd shit below...
    # > TrackPipeline       Here...
    # > DistancePipeline    Here...

    # 5. Remove file once it has been used.
    # os.remove(filepath)

    # 6) Send JSON back
    # return jsonify(...)

if __name__ == '__main__':
    app.run(debug=True)



"""Code for Mocking Shit


    #  MOCK RECOMMENDATION RETURN
    recommendations = [
      {'artist': 'burial',            'album': 'untrue',   'track': 'untrue'},
      {'artist': 'bladee',            'album': 'gluee',    'track': 'unreal'},
      {'artist': 'ecco2k',            'album': 'e',        'track': 'cc'},
      {'artist': 'against all logic', 'album': '2012-2017','track': 'i never dream'},
      {'artist': 'playboi carti',     'album': 'music',    'track': 'opm babi'},
    ]

    return jsonify(recommendations)
"""