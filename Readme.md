<h1 align="center">🔒 CNN + Multimodal Malicious Website Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Research%20Project-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-Kaggle%20Phishing%20Screenshots-yellow?style=flat-square"/>
</p>

---

### 🧩 Overview
> A **Deep Learning–based cybersecurity system** that identifies **malicious or phishing websites** by analyzing webpage **screenshots and URLs**.  
> This project uses a **Convolutional Neural Network (CNN)** combined with **lexical (URL text) features** in a **multimodal fusion** model to detect phishing attempts with over **97% accuracy**.

📖 *Inspired by the research paper:*  
**“CNN Based Malicious Website Detection by Invalidating Multiple Web Spams”**

---

### 🌟 Highlights
✅ Screenshot-based **CNN model** for visual webpage analysis  
✅ **Multimodal fusion** (CNN + URL lexical features)  
✅ Explainable AI with **Grad-CAM visualizations**  
✅ **Classical ML comparisons** — RandomForest, SVM, Logistic Regression  
✅ Beautiful **ROC, F1, and Confusion Matrix visualizations**  
✅ Fully **Google Colab compatible**  

---

### 🧠 System Architecture

mathematica
Copy code
     ┌───────────────────────────────┐
     │          Web Input            │
     └───────────────────────────────┘
                    │
       ┌─────────────────────────┐
       │   Screenshot Image       │
       └─────────────────────────┘
                    │
                    ▼
      ┌───────────────────────────┐
      │   CNN Feature Extractor   │
      └───────────────────────────┘
                    │
┌───────────────────────┐ │
│ URL Text Features │──TF-IDF───────┘
└───────────────────────┘
│
▼
┌─────────────────────────────┐
│ Fusion Dense Classifier │
└─────────────────────────────┘
│
▼
⚡ Output: {Benign | Malicious}

yaml
Copy code

---

### ⚙️ Model Design

| Component | Description |
|------------|-------------|
| **CNN Backbone** | 3 Convolution + MaxPool layers (input 256×256×3) |
| **Lexical Branch** | TF-IDF (char 2–5 n-grams) + URL statistics |
| **Fusion Network** | Dense(128 → 64) + Dropout |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Metrics** | Accuracy, F1, Precision, Recall, AUC |

---

### 🗂 Dataset

**📦 Source:** [Kaggle – Phishing Sites Screenshot Dataset](https://www.kaggle.com/datasets/zackyzac/phishing-sites-screenshot)

| Type | Count | Folder |
|------|--------|--------|
| ✅ Legitimate | ~1000 | `/dataset/legitimate/` |
| ⚠️ Phishing | ~1000 | `/dataset/phishing/` |

All data consists of *webpage screenshots*, labeled `0` for benign and `1` for phishing.

---

### 🧰 Setup Instructions (Google Colab)

#### 1️⃣ Clone / Upload Project
```bash
!git clone https://github.com/AYUSH-GANGWAR9/Malicious-Website-Detection.git
%cd Malicious-Website-Detection
2️⃣ Install Dependencies
bash
Copy code
!apt-get update -qq
!apt-get install -y -qq chromium-browser
!pip install -q selenium webdriver-manager pillow pandas matplotlib scikit-learn tensorflow==2.12.0 kaggle seaborn
3️⃣ Setup Kaggle API
python
Copy code
import os
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"
4️⃣ Run Notebook Cells
Open the notebook in Colab (malicious_detection.ipynb) and execute cells sequentially:

Download dataset

Generate labels.csv

Train CNN model

Train multimodal model

Visualize and evaluate

📈 Results & Performance
Model	Accuracy	Precision	Recall	F1	AUC
CNN (Image Only)	94.7%	94%	93%	0.94	0.95
Multimodal (Image + URL)	97.6%	97%	97%	0.97	0.98

📊 Visualization Outputs
Confusion Matrix

<p align="center"> <img src="assets/confusion_matrix.png" width="400"/> </p>
ROC Curve

<p align="center"> <img src="assets/roc_curve.png" width="400"/> </p>
Grad-CAM Explanation

<p align="center"> <img src="assets/gradcam.png" width="400"/> </p>
⚖️ Classical ML Baseline Comparison
Model	Accuracy	F1
Logistic Regression	89.2%	0.89
Random Forest	92.4%	0.92
SVM (RBF)	90.1%	0.90
Multimodal CNN (Ours)	97.6%	0.97

🎨 Explainability (Grad-CAM)
Grad-CAM highlights the regions in webpage screenshots most responsible for predicting phishing, such as fake login prompts or suspicious input forms.

This step adds transparency and interpretability to deep learning cybersecurity models.

🔮 Future Work
Feature	Description
🛰 Streamlit Dashboard	Real-time interface with risk-level visualization
⚡ TF Lite / ONNX Conversion	Edge deployment for lightweight models
🔐 Adversarial Robustness	Detect obfuscated phishing pages
📊 Ablation Studies	Compare CNN-only, lexical-only, and fusion models

🧠 Technologies Used
Python 3.10

TensorFlow / Keras

Scikit-learn

Pandas / NumPy

Matplotlib / Seaborn

Kaggle API Integration

Grad-CAM Explainability

📁 Project Structure
bash
Copy code
📂 Malicious-Website-Detection/
 ├── malicious_detection.ipynb    # Full Colab notebook
 ├── README.md                    # Documentation
 ├── /dataset/                    # Kaggle dataset (auto-downloaded)
 ├── cnn_base.h5                  # Trained CNN model
 ├── multimodal_model.h5          # Trained fusion model
 ├── /assets/                     # Visualizations (Grad-CAM, ROC, CM)
 └── requirements.txt             # Dependencies (optional)
🧑‍💻 Author
👨‍💻 Ayush Gangwar
Machine Learning | Deep Learning | Cybersecurity Research Enthusiast

📫 Connect with me:

🔗 LinkedIn: linkedin.com/in/911ayushgangwar

💻 GitHub: github.com/AYUSH-GANGWAR9

<h3 align="center">⭐ If you find this project helpful, consider giving it a star on GitHub!</h3> ```