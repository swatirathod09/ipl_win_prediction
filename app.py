"""Flask server for the IPL Win Probability Predictor.
Run:  python app.py     then open http://127.0.0.1:5000"""
import json
import sys
from pathlib import Path
import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
import inference

bundle = joblib.load(ROOT / "model" / "win_model.joblib")
metrics = json.loads((ROOT / "model" / "metrics.json").read_text())
probability = inference.model_probability(bundle)

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="")

@app.get("/")
def home():
    return app.send_static_file("index.html")

@app.get("/api/options")
def options():
    return jsonify(teams=bundle["teams"], venues=bundle["venues"], metrics=metrics)

@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    if data.get("batting_team") not in bundle["teams"] or data.get("bowling_team") not in bundle["teams"]:
        return jsonify(error="Choose a batting team and a bowling team."), 400
    if data.get("venue") not in bundle["venues"]:
        return jsonify(error="Choose a venue."), 400
    try:
        return jsonify(inference.situation(data, probability))
    except ValueError as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
