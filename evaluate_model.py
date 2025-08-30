import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Config - adjust if needed
MODEL_PATH = "speech_emotion.h5"
FEATURES = "features.npy"
LABELS = "labels.npy"
CLASSES_FILE = "classes.npy"
N_MFCC = 40
MAX_PAD_LEN = 174

# Load
model = tf.keras.models.load_model(MODEL_PATH)
X = np.load(FEATURES)
y = np.load(LABELS)
classes = np.load(CLASSES_FILE, allow_pickle=True) if os.path.exists(CLASSES_FILE) else None

# Train/test split (use same random_state you used during training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reshape
X_test = X_test.reshape(X_test.shape[0], N_MFCC, MAX_PAD_LEN, 1)

# one-hot for report
y_test_cat = to_categorical(y_test, num_classes=len(np.unique(y)))

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat, verbose=1)
print("Test loss:", loss, "Test acc:", acc)

# Predictions
preds = model.predict(X_test)
y_pred = preds.argmax(axis=1)

# Report
if classes is None:
    class_names = [str(i) for i in range(len(np.unique(y)))]
else:
    class_names = list(classes)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
