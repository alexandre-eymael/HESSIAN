from flask import Flask, render_template, request
from utils import parse_uploaded_image
import pathlib
import json
from database.database import HessianDatabase
#from models.inference import load_model, predict_image

###### Web Server & Preprocessing

# Folders
UPLOAD_FOLDER = "/tmp/hessian_uploads"
STORAGE_PATH = "./storage"

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
# inference_models = {model_id : load_model(model_name) for model_id, model_name, _, _ in models}

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

def _api(api_key, image, model_id):

    user = db.get_user_from_api_key(api_key)
    if not user:
        return {"error" : "Invalid API key"}, 401
    
    try:
        result = _inference(image, model_id)
    except Exception as e:
        return {"error" : str(e)}, 500
        
    return result, 200

@app.route("/api", methods=["GET"])
def api():

    api_key = request.headers.get("HESSIAN-API-Key")
    image = request.json.get("image")
    model_id = request.json.get("model_id")

    return _api(api_key, image, model_id)

###### API Frontend

@app.route("/")
def submit():
    return render_template(
        "submit.html",
        models = {model_id : model_name for model_id, model_name, price, version in models}
    )

@app.route("/result", methods=["POST"])
def results():

    api_key = request.form.get("api_key")
    query = request.files.get("query")
    model_id = request.form.get("model_id")
    image = parse_uploaded_image(app, query)

    predictions, status_code = _api(api_key, image, model_id)

    if status_code != 200:
        error = predictions.get("error", "Unknown error")
        return render_template(
            "error.html",
            error_message = error
        )

    db.add_query(image, api_key, model_id)

    return render_template(
        "results.html",
        base_image = image,
        predictions = {' '.join(cls_name.replace('_', ' ').split()): proba*100 for cls_name, proba in predictions.items()}
    )