import os ######
import subprocess
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/regression", methods=["POST"])
def regression():
    file = request.files["dataset"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    subprocess.run(["python", "scripts/generate_regression.py", path])
    return render_template("regression.html")

@app.route("/classification", methods=["POST"])
def classification():
    file = request.files["dataset"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    subprocess.run(["python", "scripts/generate_classification.py", path])
    return render_template("classification.html")

if __name__ == "__main__":
    app.run(debug=True)