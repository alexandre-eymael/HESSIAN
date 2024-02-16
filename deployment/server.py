from flask import Flask, render_template, request
from utils import parse_uploaded_image
import json

###### Web Server & Preprocessing

UPLOAD_FOLDER = "/tmp/hessian_uploads"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open("db.json", "r") as f:
    db = json.load(f)

###### API
def _inference():
    pass

def _api(api_key, image, model_size):

    # user = db.get(api_key)
    # if user is None:
    #     return {"error" : "Invalid API key"}, 401

    # Perform inference
    # ...
    _inference()
    result = {"cat" : 0.85, "dog" : 0.05}, 200

    return result

@app.route("/api", methods=["GET"])
def api():

    api_key = request.headers.get("HESSIAN-API-Key")
    image = request.json.get("image")
    model_size = request.json.get("model_size")

    return _api(api_key, image, model_size)

###### API Frontend

@app.route("/")
def submit():
    return render_template("submit.html")

@app.route("/result", methods=["POST"])
def results():

    api_key = request.form.get("api_key")
    query = request.files.get("query")
    model_size = request.form.get("model_size")
    image = parse_uploaded_image(app, query)

    predictions, status_code = _api(api_key, image, model_size)

    return render_template(
        "results.html",
        base_image=image,
        predictions={k.capitalize() : v*100 for k,v in predictions.items()}
    )