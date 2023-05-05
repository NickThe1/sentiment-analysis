import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

app = Flask(__name__)


model_path = "sentiment_model.pt"
model = torch.jit.load(model_path)

def preprocess(text):
    text = tokenize_and_clear(text)
    return text

@app.route("/api/sentiment", methods=["POST"])
def predict_sentiment():
    text = request.json["text"]
    text = preprocess(text)

    with torch.no_grad():
        output = model(text)

    probs = F.softmax(output, dim=0)
    pred_label = torch.argmax(probs).item()
    confidence = probs[pred_label].item()

    if pred_label == 1:
        pred_string = "positive"
    else:
        pred_string = "negative"

    response = {"sentiment": pred_string, "confidence": confidence}
    return jsonify(response)