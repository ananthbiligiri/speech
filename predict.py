import os
import sys
import numpy as np
import librosa
import tensorflow as tf

# ---------- Config ----------
MODEL_FILES = ["speech_emotion_bilstm.h5", "speech_emotion.h5", "speech_emotion_model.h5"]
MODEL_PATH = next((m for m in MODEL_FILES if os.path.exists(m)), None)
CLASSES_FILE = "classes.npy"   # saved during extract_features.py
SR = 22050
N_MFCC = 40
MAX_PAD_LEN = 174
# ----------------------------

if MODEL_PATH is None:
    raise FileNotFoundError("Model file not found. Place your .h5 model in the project folder.")

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# Load class labels saved during feature extraction
if os.path.exists(CLASSES_FILE):
    emotion_labels = list(np.load(CLASSES_FILE, allow_pickle=True))
    print("Loaded classes from", CLASSES_FILE, "->", emotion_labels)
else:
    # Fallback: user-provided list (must exactly match training order)
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad", "surprise"]
    print("WARNING: classes.npy not found. Using fallback labels (ensure this order matches training):", emotion_labels)

def extract_features(file_path, sr=SR, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
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

def predict_emotion(file_path, top_k=3):
    features = extract_features(file_path)
    if features is None:
        return None, None, None
    # model expects shape (batch, n_mfcc, frames, 1)
    features = features.reshape(1, N_MFCC, MAX_PAD_LEN, 1).astype(np.float32)

    preds = model.predict(features)[0]   # shape (num_classes,)
    top_idx = preds.argsort()[-top_k:][::-1]
    top_labels = [(emotion_labels[i], float(preds[i])) for i in top_idx]
    return emotion_labels[np.argmax(preds)] if len(emotion_labels) == preds.shape[0] else "Unknown", preds, top_labels

if __name__ == "__main__":
    # Get path from command-line or interactive input
    if len(sys.argv) >= 2:
        test_file = sys.argv[1]
    else:
        test_file = input("Enter path to audio file (.wav): ").strip()

    if not os.path.exists(test_file):
        print("File not found:", test_file)
        sys.exit(1)

    print("Predicting for:", test_file)
    label, probs, topk = predict_emotion(test_file)
    if label is None:
        print("Prediction failed.")
    else:
        print("Top prediction:", label)
        print("Top-k:", topk)
        print("All probs:", probs)
