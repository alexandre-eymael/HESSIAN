from .utils import parse_uploaded_image
from .database.database import HessianDatabase
from models.inference import load_model, predict_image

from flask import Flask, render_template, request, jsonify
import pathlib
from waitress import serve

###### Web Server & Preprocessing

HOST = "127.0.0.1"
PORT = 8000

# Folders
UPLOAD_FOLDER = "/tmp/hessian_uploads"
STORAGE_PATH = "./deployment/storage"

# Files
SCHEMA_FILE = f"{STORAGE_PATH}/schema.sql"
DATA_FILE = f"{STORAGE_PATH}/data.sql"
DB_FILE = f"{STORAGE_PATH}/hessian.db"

pathlib.Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

# Database
db = HessianDatabase(DB_FILE)
db.init_if_empty(SCHEMA_FILE, DATA_FILE)

# Load models into memory
models = db.get_models()
inference_models = {str(model_id) : load_model(model_name) for model_id, model_name, _, _ in models}

# Start Webserver
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

###### API
def _inference(image, model_id):
    try:
        model = inference_models[model_id]
    except KeyError:
        raise ValueError(f"Model type {model_id} not found")
    
    return predict_image(model, image)

def _api(api_key, image, model_name):

    user = db.get_user_from_api_key(api_key)
    if not user:
        return {"error" : "Invalid API key"}, 401
    
    model_id = db.get_model_id_by_name(model_name)
    if not model_id:
        return {"error" : "Invalid model name"}, 400

    if image is None:
        return {"error" : "Invalid image provided"}, 400

    try:
        result = _inference(image, model_id)
    except Exception as e:
        return {"error" : str(e)}, 500
        
    return result, 200

@app.route("/api")
def api():

    api_key = request.headers.get("HESSIAN-API-Key")
    image = request.args.get("image")
    model_name = request.args.get("model")

    response, status = _api(api_key, image, model_name)
    return jsonify(response)

###### API Frontend

@app.route("/")
def submit():
    return render_template(
        "submit.html",
        models = {model_id : model_name for model_id, model_name, _, _ in models}
    )

@app.route("/result", methods=["POST"])
def results():

    api_key = request.form.get("api_key")
    query = request.files.get("query")
    model_id = request.form.get("model_id")
    encoded_image, str_image = parse_uploaded_image(app, query)

    predictions, status_code = _api(api_key, encoded_image, model_id)

    if status_code != 200:
        error = predictions.get("error", "Unknown error")
        return render_template(
            "error.html",
            error_message = error
        )

    db.add_query(str_image, api_key, model_id)

    # Add 😊 if plant is healthy, ☠️ if not
    predictions = {f"{'😊' if 'healthy' in cls_name else '☠️'} {cls_name}": proba for cls_name, proba in predictions.items()}

    # Determine binary probability
    healthy_prob = round(sum([proba for cls_name, proba in predictions.items() if "healthy" in cls_name]), 2)
    sick_prob = 1. - healthy_prob

    # Keep only probabilities > 0.05, and add a "Other" class with the sum of the rest
    predictions = {cls_name: proba for cls_name, proba in predictions.items() if proba > 0.05}
    predictions["Other"] = 1 - sum(predictions.values())

    # Parse predictions for display
    predictions = {' '.join(cls_name.replace('_', ' ').split()): round(proba*100, 2) for cls_name, proba in predictions.items()}

    return render_template(
        "results.html",
        base_image = str_image,
        predictions = predictions,
        healthy_prob = healthy_prob * 100,
        sick_prob = sick_prob * 100
    )

## Serve
if __name__ == '__main__':
    print(f"Starting server at {HOST}:{PORT}")
    serve(app, host=HOST, port=PORT, threads=2, connection_limit=100)