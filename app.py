import os
import sys
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/regression", methods=["POST"])
def regression():
    file = request.files["dataset"]
    filename = secure_filename(file.filename)

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    cmd = f'{sys.executable} scripts/generate_regression.py "{file_path}"'
    os.system(cmd)

    # read metrics file
    metrics_file = "static/graphs/regression/metrics.txt"
    metrics = open(metrics_file).read() if os.path.exists(metrics_file) else ""

    return render_template("regression.html", metrics=metrics)


@app.route("/classification", methods=["POST"])
def classification():
    file = request.files["dataset"]
    filename = secure_filename(file.filename)

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    cmd = f'{sys.executable} scripts/generate_classification.py "{file_path}"'
    os.system(cmd)

    metrics_file = "static/graphs/classification/metrics.txt"
    metrics = open(metrics_file).read() if os.path.exists(metrics_file) else ""

    return render_template("classification.html", metrics=metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)