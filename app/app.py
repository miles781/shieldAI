"""
app.py

The web layer of the project. Two routes:
- "/"          serves the HTML page
- "/predict"   receives text from the browser, runs it through the
               existing moderate_text() pipeline, and returns JSON

This file does NOT contain any ML logic itself - it just calls the
functions we already built and tested in src/moderation.py.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.moderation import moderate_text

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    result = moderate_text(text)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
