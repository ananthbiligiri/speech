import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "data"

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

# Build dataset
X, y = [], []
labels = os.listdir(DATASET_PATH)

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(label)

X = np.array(X)
y = np.array(y)

# 🔥 Encode labels (angry, happy, etc.) -> integers (0,1,2…)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("✅ Features extracted:", X.shape)
print("✅ Labels extracted:", y.shape)
print("✅ Classes found:", list(label_encoder.classes_))

# Save features + encoded labels
np.save("features.npy", X)
np.save("labels.npy", y_encoded)
np.save("classes.npy", label_encoder.classes_)  # optional, to know which index = which emotion

print("✅ Saved features.npy, labels.npy and classes.npy")
