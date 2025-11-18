<h1 align="center">🔒 CNN + Multimodal Malicious Website Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-Kaggle%20Phishing%20Screenshots-yellow?style=flat-square"/>
</p>

---

### 🧩 Overview
> A **Deep Learning-based cybersecurity system** that identifies *malicious or phishing websites* using both **visual webpage screenshots** and **URL-based lexical patterns**.  
> This project fine-tunes a **ResNet-50 CNN** and fuses its features with **TF-IDF lexical embeddings** to detect phishing attempts with **~97–98% accuracy**.

Inspired by:  
📄 *“CNN-Based Malicious Website Detection by Invalidating Multiple Web Spams”* (IEEE, 2023)

---

### 🌟 Key Features
✅ Fine-tuned **ResNet-50 backbone** (transfer learning from ImageNet)  
✅ **Multimodal Fusion** — combines screenshot + URL features  
✅ **Grad-CAM Explainability** for visual insights  
✅ **Dark-Themed Streamlit Dashboard** with live “Risk Meter”  
✅ **Fully runnable on Google Colab** (Cloudflare tunnel for public demo link)  
✅ **Classical ML baselines** (SVM, RandomForest, LogisticRegression)

---

### 🧠 Model Summary

| Component | Description |
|------------|-------------|
| **Visual Backbone** | Fine-tuned ResNet-50 pretrained on ImageNet |
| **Lexical Branch** | TF-IDF + statistical features (length, digits, specials) |
| **Fusion Layer** | Concatenation + Dense(128 → 64 → 1) |
| **Loss Function** | Binary Cross-Entropy |
| **Optimizer** | Adam (1e-4 → 1e-5 during fine-tuning) |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC |

---

### 📦 Dataset

**Source:** [Kaggle – Phishing Sites Screenshot Dataset](https://www.kaggle.com/datasets/zackyzac/phishing-sites-screenshot)

| Category | Count | Folder |
|-----------|--------|--------|
| ✅ Legitimate | ~1000 | `/dataset/legitimate/` |
| ⚠️ Phishing | ~1000 | `/dataset/phishing/` |

Each sample is a webpage **screenshot**, labeled as 0 (benign) or 1 (phishing).

---

### 🧰 Setup & Execution (Google Colab)

#### 1️⃣ Clone Repository
```bash
!git clone https://github.com/AYUSH-GANGWAR9/Malicious-Website-Detection.git
%cd Malicious-Website-Detection
2️⃣ Install Dependencies
bash
Copy code
!pip install -q tensorflow==2.12.0 keras scikit-learn pillow pandas seaborn matplotlib streamlit cloudflared kaggle
3️⃣ Run Notebook
Open malicious_detection.ipynb in Colab and run all cells in order:

Download Dataset

Generate labels.csv (with synthetic URLs)

Train ResNet-50

Fine-tune (Phase 2)

Train Multimodal Fusion Model

Evaluate + Visualize

Results & Performance
Model	Accuracy	F1	AUC
Baseline CNN	70 %	0.69	0.72
Fine-Tuned ResNet-50	93 – 95 %	0.94	0.96
Multimodal Fusion (CNN + URL)	97 – 98 %	0.97 +	0.98 +
📊 Visual Outputs

Confusion Matrix

<p align="center"><img src="assets/confusion_matrix.png" width="400"/></p>

ROC Curve

<p align="center"><img src="assets/roc_curve.png" width="400"/></p>

Grad-CAM Visualization

<p align="center"><img src="assets/gradcam.png" width="400"/></p>
⚖️ Classical ML Baselines
Model	Accuracy	F1
Logistic Regression	91 %	0.90
Random Forest	94 %	0.93
SVM (RBF)	93 %	0.92
Multimodal CNN (Ours)	97 %	0.97
🎓 Research Highlights

“The combination of deep visual understanding from screenshots and lexical URL patterns offers superior detection performance compared to single-modality approaches.”

Future Enhancements
Feature	Description
🛰 Streamlit Cloud Deployment	Host dashboard permanently
⚡ TF-Lite / ONNX Conversion	Edge/browser plugin inference
🔐 Adversarial Defense	Handle obfuscated phishing URLs
📊 Ablation Study	Compare CNN-only vs URL-only vs Fusion
🧾 Automated Report Generator	Generate IEEE-style project report
💻 Technologies Used

Python 3.10 · TensorFlow 2.12 · Keras

Scikit-learn · Pandas · NumPy

Matplotlib · Seaborn · Streamlit · Cloudflared

Kaggle API integration

## 🧑‍💻 Author

👨‍💻 Ayush Gangwar
Machine Learning · Deep Learning

📫 Connect with me:

🔗 LinkedIn : https://www.linkedin.com/in/911ayushgangwar/
