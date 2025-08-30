from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model and classes
model = tf.keras.models.load_model("speech_emotion.h5")
classes = np.load("classes.npy", allow_pickle=True) if os.path.exists("classes.npy") else []

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_features(file_path, sr=22050, n_mfcc=40, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=3, offset=0.5, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error extracting features:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and predict
        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Error processing audio file'})
            
        features = features.reshape(1, 40, 174, 1).astype(np.float32)
        preds = model.predict(features)[0]
        
        # Get top prediction
        pred_idx = np.argmax(preds)
        emotion = classes[pred_idx] if len(classes) > pred_idx else str(pred_idx)
        confidence = float(preds[pred_idx])
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': {str(cls): float(conf) for cls, conf in enumerate(preds)}
        })

if __name__ == '__main__':
    app.run(debug=True)