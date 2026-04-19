import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from predict_pipeline import predPipeline

# NOTE: flask_cors is added to allow the frontend (index.html opened directly in the
# browser) to make requests to this Flask server. Without this, the browser blocks
# cross-origin requests due to CORS policy, resulting in a network error on the frontend.
from flask_cors import CORS

os.chdir(os.path.dirname(os.path.abspath(__file__)))

res = {"Class": "", "Status": ""}

app = Flask(__name__)

# NOTE: CORS(app) enables cross-origin requests for all routes in the app.
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'

with open("data/threshold_setting.json", "r") as file:
    threshold = json.load(file)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        pred = predPipeline(filename)

        res['Class'] = f"{pred}"

        # NOTE: Using threshold from data/threshold_setting.json instead of exact 0.0
        # comparison. The model outputs a float probability between 0 and 1.
        # Checking pred == 0.0 would never match, flagging everything as Fraud.
        if pred < threshold['threshold']:
            res['Status'] = 'Legitimate'
        else:
            res['Status'] = 'Fraud'
        
        return jsonify(res), 200

if __name__ == '__main__':
    app.run(debug=True)