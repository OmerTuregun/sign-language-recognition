## Sign Language Recognition System ✋🤟

## 📌 Project Overview

This project focuses on building a real-time sign language recognition system using deep learning and computer vision.
A custom dataset was collected via webcam, augmented with image processing techniques, and trained with a Convolutional Neural Network (CNN) to recognize sign language letters.

The system works in real-time, detecting hand signs from the camera and converting them into recognized letters on-screen.
This project aims to improve accessibility by enabling communication between sign language users and non-sign language speakers.


## 🚀 Features

- Custom Dataset Collection (captured using OpenCV)
- Data Augmentation with Albumentations (rotation, brightness, scaling, flipping)
- CNN Model Training with TensorFlow/Keras
- Model Optimization using EarlyStopping & ReduceLROnPlateau
- Data Preprocessing (train/val/test split, normalization, batch loading)
- Real-Time Prediction with OpenCV live camera feed
- High Accuracy on both validation and test sets


## 📂 Project Structure

sign-language-recognition/
│── veri_olustur.py          # Dataset creation via webcam
│── veri_zenginlestir.py     # Data augmentation (Albumentations)
│── veri_bolme.py            # Train/Val/Test split
│── tensorflow_egitim.py     # CNN training with TensorFlow/Keras
│── test.py                  # Real-time prediction with webcam
│── dataset/                 # Raw dataset (not uploaded)
│── augmented_dataset/       # Augmented images (not uploaded)
│── prepared_dataset/        # Final dataset split (not uploaded)
│── sign_language_model.h5   # Trained model (saved)
│── README.md                # Project description


## 📊 Model Architecture

- Input: 128x128 RGB images
- Layers: Conv2D + MaxPooling + Dropout + Dense
- Output: 26 classes (A–Z letters)
- Optimizer: Adam (lr=0.0005)
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy


## 📈 Results

- Training Accuracy: ~95%
- Validation Accuracy: ~90%
- Real-time prediction works with smooth performance


## 📦 Installation

# Clone the repository
git clone https://github.com/OmerTuregun/sign-language-recognition.git
cd sign-language-recognition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt


## ▶️ Usage

1. Collect Dataset
python veri_olustur.py

2. Augment Dataset
python veri_zenginlestir.py

3. Split Dataset
python veri_bolme.py

4. Train Model
python tensorflow_egitim.py

5. Run Real-Time Prediction
python test.py


## 📥 Dataset

Due to file size limitations, datasets are not included in this repository.
You can download them here:

🔗 Kaggle Dataset Link (replace with your link)


## 🛠 Technologies & Libraries

Python
TensorFlow / Keras
OpenCV
Albumentations
NumPy, Matplotlib


## 👨‍💻 Authors

Ömer Faruk Türegün


## 📜 License

This project is licensed under the MIT License.