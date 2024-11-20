# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from utils.preprocessing import preprocess_video

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the TensorFlow SavedModel
MODEL_DIR = 'saved_movinet_model'  # Path to your SavedModel directory
print(f"Loading model from: {MODEL_DIR}")
model = tf.saved_model.load(MODEL_DIR)

# List all available signatures (for debugging purposes)
print("Available Signatures:")
for key in model.signatures.keys():
    print(f" - {key}")

# Choose the 'serving_default' signature for inference
infer = model.signatures['serving_default']  # Using 'serving_default'

# Define class labels
labels = [
    'ADVANTAGE', 'ALASKA', 'ALLOFSUDDEN', 'ANYONE', 'APPLE', 'BACKPACK1', 'BECOME', 'BETWEEN', 'BREAKFAST1',
    'BULLSHIT', 'CAPTURE', 'CASTLE2', 'CELEBRATE', 'CEMETERY', 'CIRCUSWHEEL', 'CLIENT', 'COMMUNITY', 'COOL3',
    'CORNER', 'COUNT', 'DOG1', 'DONTFEELLIKE', 'DOWNHILL', 'EACHOTHER', 'EASYTODO', 'EMPTY2', 'EVERYNIGHT',
    'EXPERIMENT', 'FAIL', 'FASHION', 'FLOAT1', 'GETTOGETHER', 'GOODYGOODYSHOEOPPOSITE', 'GREET1', 'GYM',
    'HATBRIM', 'HUG', 'IMPOSSIBLE', 'IMPROVE', 'INFLUENCE', 'INSTITUTE', 'JEWELRY', 'JOKE', 'JUICE1', 'KNIGHT1',
    'KNOCKOFF', 'LABEL', 'LAYDOWN', 'LEND', 'LICKENVELOPE1', 'LONGAGO', 'MAPLE', 'MOSQUITO', 'NICE', 'NOSE',
    'PACK', 'PARANOID', 'PARK', 'PEACOCK', 'PEEL1', 'PERCEIVE', 'PONDER', 'POWDER', 'PRECIOUS', 'PROPOSE',
    'PULLCONVINCE', 'REMOVE', 'ROSE', 'SCOLD', 'SHORTWORD', 'SIGNPAPER', 'SMOOTH', 'SOCCER2', 'SOCIAL',
    'SOCIETY', 'SOPHOMORE', 'SPICY', 'SQUARE', 'SQUEEZE', 'STINK', 'STORY1', 'STUMBLE', 'SUPERVISOR',
    'SWEARIN', 'THANKYOU', 'THEORY', 'THERAPY', 'TOSS', 'TOSSOUT', 'TOTAL', 'TOY1', 'TUBE', 'TURN', 'UNCERTAIN',
    'WATERDROP', 'WE', 'WEAVE', 'WHISPER', 'WIDE', 'WILLGO'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        video_bytes = file.read()
        preprocessed_frames = preprocess_video(video_bytes)

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(preprocessed_frames, dtype=tf.float32)

        # Prepare input as a dictionary matching the signature's input key
        inputs = {'image': input_tensor}

        # Perform inference
        outputs = infer(**inputs)

        # Debug: Print available output keys
        print("Available output keys:", outputs.keys())

        # Access the output tensor using the correct output key
        logits = outputs['classifier_head_1']  # Updated output key

        # Apply softmax to get probabilities
        predictions = tf.nn.softmax(logits, axis=-1)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = predictions.numpy()[0][predicted_class]

        response = {
            'predicted_label': labels[predicted_class],
            'confidence': float(confidence)
        }
        return jsonify(response)

    except Exception as e:
        # For debugging purposes, log the exception
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Sign Language Recognition API is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
