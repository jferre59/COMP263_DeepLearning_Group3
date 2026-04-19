import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from predict_pipeline import predPipeline

os.chdir(os.path.dirname(os.path.abspath(__file__)))

res = {"Class": "", "Status": ""}

app = Flask(__name__)
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

        if pred < threshold['threshold']:
            res['Status'] = 'Legitimate'
        else:
            res['Status'] = 'Fraud'
        
        return jsonify(res), 200

if __name__ == '__main__':
    app.run(debug=True)