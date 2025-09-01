# 🎤 Speech Emotion Recognition System

This project is a **Speech Emotion Recognition (SER) system** that classifies emotions from human voice/audio using **Deep Learning** and **TensorFlow/Keras**.  
The model can detect emotions such as **happy, sad, angry, neutral, fear, surprise**, etc.  

---

## 🚀 Features
- Preprocessing of speech signals (MFCCs, spectrograms, etc.)
- Deep Learning model trained on emotion datasets
- Real-time emotion detection from microphone input
- Supports multiple emotion classes
- Visualization of training accuracy & loss

---

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍  
- **Libraries**: TensorFlow, Keras, NumPy, Librosa, Matplotlib, Scikit-learn  
- **Model Type**: CNN / RNN / LSTM (depending on your implementation)  

---

## 📂 Project Structure
      speech-emotion-recognition/
      │-- data/ # Dataset (not included in repo)
      │-- models/ # Saved model files (.h5)
      │-- notebooks/ # Jupyter notebooks for experiments
      │-- src/ # Source code
      │ ├── preprocess.py # Audio preprocessing
      │ ├── train.py # Model training
      │ ├── predict.py # Prediction script
      │ └── app.py # Streamlit / Flask app
      │-- requirements.txt # Dependencies
      │-- README.md # Project documentation


---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ananthbiligiri/speech.git
   cd speech



Create a virtual environment:

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

## Output
<img width="1059" height="510" alt="Image" src="https://github.com/user-attachments/assets/bfb7a60d-b231-4893-a678-cd800007959c" />

## ▶️ Usage
Train the model
python src/train.py

## 🔮 Future Improvements
Improve accuracy with larger datasets

📜 License
This project is licensed under the MIT License.

##   👨‍💻 Author
| Ananth B S |

