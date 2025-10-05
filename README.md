# Deep-learning-model



COMPANY : CODTECH IT SOLUTIONS

NAME : RAHINI SR

INTERN ID : CT04DY2633

DOMAIN NAME : DATA SCIENCE

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH

# 👗 Fashion Image Classification using Deep Learning

## 📌 Overview

This project implements a **Deep Learning model** for **image classification** using the **Fashion-MNIST dataset**.  
The system automatically identifies various fashion items such as T-shirts, shoes, and bags through a **Convolutional Neural Network (CNN)** built with **TensorFlow & Keras**.

The model was trained on 60,000 images and tested on 10,000 unseen samples, achieving high accuracy through multiple convolutional and pooling layers.

---

## ⚙️ Features

* ✅ Image preprocessing (normalization & reshaping)  
* ✅ CNN model built with TensorFlow/Keras  
* ✅ Visualization of **accuracy**, **loss**, and **confusion matrix**  
* ✅ Random test sample predictions with true vs predicted labels  
* ✅ Model saving and reloading for reuse (`.h5` format)

---

## 🧠 Model Architecture

| Layer Type | Parameters | Output Shape |
|-------------|-------------|---------------|
| Conv2D | 32 filters, 3x3 kernel, ReLU | (26, 26, 32) |
| MaxPooling2D | 2x2 | (13, 13, 32) |
| Conv2D | 64 filters, 3x3 kernel, ReLU | (11, 11, 64) |
| MaxPooling2D | 2x2 | (5, 5, 64) |
| Flatten | — | (1600) |
| Dense | 128 units, ReLU | (128) |
| Dense | 10 units, Softmax | (10) |

---

## 🛠 Tools & Technologies

* **Editor:** Visual Studio Code  
* **Language:** Python 3.11  
* **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn  

---

## 📊 Performance & Results

| Metric | Result |
|--------|---------|
| **Test Accuracy** | ~91% |
| **Loss Function** | Sparse Categorical Cross-Entropy |
| **Optimizer** | Adam |

### 📈 Visualization Samples
- Training & Validation Accuracy Curve  
- Training & Validation Loss Curve  
- Confusion Matrix Heatmap  
- Random Sample Predictions  

---

## 📂 Project Structure

fashion-image-classification/
│── data/                  # Dataset (auto-downloaded from Keras)
│── models/                # Saved CNN model (.h5)
│── outputs/               # Accuracy & confusion matrix plots
│── src/                   # Source code (main.py)
│── README.md              # Project documentation
│── requirements.txt       # Dependencies

🖼️ Random Predictions:

| True Label | Predicted Label |
|-------------|-----------------|
| Dress | Dress ✅ |
| Bag | Bag ✅ |
| Shirt | Coat ❌ |

---

## Output 

<img width="1920" height="988" alt="Image" src="https://github.com/user-attachments/assets/28f7f60e-4bdf-4c90-87c3-46a692e64d60" />

<img width="1920" height="983" alt="Image" src="https://github.com/user-attachments/assets/c3088e85-1c66-4edc-acd4-4e68b3487aa9" />

<img width="1920" height="980" alt="Image" src="https://github.com/user-attachments/assets/daa27ee6-ad30-4f2b-9724-f58e92a84e22" />

<img width="1920" height="988" alt="Image" src="https://github.com/user-attachments/assets/e2f328e4-6de3-40fa-ba08-a842af9cdf3a" />

<img width="1097" height="524" alt="Image" src="https://github.com/user-attachments/assets/b3550946-3ef4-4c0b-9231-80a76c8327e3" />

---
