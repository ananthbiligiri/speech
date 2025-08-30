import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
    Bidirectional, LSTM, BatchNormalization, Reshape
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ----------------------------
# Load features & labels
# ----------------------------
X = np.load("features.npy")   # Features saved from extract_features.py
y = np.load("labels.npy")     # Labels saved from extract_features.py

# ----------------------------
# One-hot encode labels
# ----------------------------
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Reshape input for CNN (add channel dimension)
# ----------------------------
X_train = X_train[..., np.newaxis]   # shape: (samples, time, features, 1)
X_test = X_test[..., np.newaxis]

# ----------------------------
# Model architecture
# ----------------------------
model = Sequential()

# CNN feature extractor
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

# Flatten into sequences for RNN
model.add(Reshape((X_train.shape[1] // 4, -1)))  # adjust sequence length after pooling

# BiLSTM layers
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))

# Dense classifier
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# ----------------------------
# Compile the model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

# ----------------------------
# Save model
# ----------------------------
model.save("speech_emotion.h5")
print("✅ Model trained and saved as speech_emotion_bilstm.h5")
