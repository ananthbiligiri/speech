import os
import shutil

# Input dataset path
DATASET_PATH = "Audio_Speech_Actors_01-24"

# Output organized dataset
OUTPUT_PATH = "data"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Emotion labels mapping
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Create emotion folders
for emotion in emotions.values():
    os.makedirs(os.path.join(OUTPUT_PATH, emotion), exist_ok=True)

# Loop through all actor folders
for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]  # 3rd element in filename
            emotion_label = emotions[emotion_code]
            
            src = os.path.join(actor_path, file)
            dst = os.path.join(OUTPUT_PATH, emotion_label, file)
            shutil.copy(src, dst)

print("✅ Dataset reorganized into emotion folders at:", OUTPUT_PATH)
