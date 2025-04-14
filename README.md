# 🌸 Flower Image Classification App

This project is a web app that classifies flower images into one of five categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The app is built with **Streamlit** for an interactive user interface.

---

## 🧠 Model Overview

The trained model classifies flowers into these categories:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

The CNN architecture includes:

- Data Augmentation (RandomFlip, RandomZoom, RandomRotation)
- Rescaling
- 3 Convolutional + MaxPooling layers
- Dropout for regularization
- Dense layers for classification

---

## 📂 Project Structure
├── app.py # Streamlit app code ├── Flower_Recog_model.h5 # Trained model file ├── tempDir/ # Temporary storage for uploaded images ├── flower_photos/ # (Ignored in GitHub using .gitignore) ├── requirements.txt # Project dependencies ├── .gitignore └── README.md


📷 How It Works
Upload a flower image.

The image is preprocessed and fed into the model.

The app shows the predicted flower type with the confidence percentage.

🧾 Example Output
Prediction: The image belongs to sunflowers with a confidence of 94.27%.

📦 Requirements
Dependencies include:

TensorFlow

Keras

Streamlit

NumPy

Pillow
