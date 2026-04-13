# 🫁 COPD Prediction System

An AI-powered early detection system for **Chronic Obstructive Pulmonary Disease (COPD)** using a dual-model fusion approach — combining deep learning on lung images with clinical spirometry data.

---

## 🤖 Models

| Model | Input | Architecture | Accuracy |
|-------|-------|-------------|----------|
| **CNN** | Lung CT/X-ray (128×128 RGB) | Conv2D → MaxPool → Dense → Sigmoid | **98.9%** |
| **ANN** | 8 clinical features | Dense(8) → Dropout → Dense(4) → Sigmoid | **~95%** |
| **Fusion** | CNN + ANN outputs | Weighted average (50/50) | — |

### Clinical Features (ANN)

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | 0 = Female, 1 = Male |
| Height_cm | Height in centimetres |
| Weight_kg | Weight in kilograms |
| BMI | Body Mass Index (auto-calculated in app) |
| Smoking_Status | 0 = Never, 1 = Former, 2 = Current |
| FEV1_L | Forced Expiratory Volume in 1 second (litres) |
| FVC_L | Forced Vital Capacity (litres) |

---

## 🚀 Quick Start

### Option A — One-click (Windows)
```
Double-click setup.bat
```

### Option B — Manual
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app/copd.py
```

App opens at **http://localhost:8501**

---

## 🖥️ App Features

- **Predict tab** — Upload lung image + enter patient data → get instant COPD prediction
  - Auto BMI calculation from height/weight
  - Live FEV1/FVC ratio preview with obstruction warning
  - GOLD-inspired severity badge (Normal / Mild / Moderate / High Risk)
  - Model breakdown with visual progress bars
  - 5-metric clinical insights panel
- **Dataset Stats tab** — Spirometry data explorer with statistics
- **About tab** — Model architecture, clinical background, project structure

---

## 🔬 How the Fusion Works

```
Lung Image ──► CNN ──► cnn_copd_prob = 1 - cnn_raw
                                              ↓
                              final = 0.5 × cnn + 0.5 × ann  ──► Result
                                              ↑
Clinical Data ──► ANN ──► ann_copd_prob = ann_raw
```

> **FEV1/FVC < 0.70** is the GOLD standard criterion for airflow obstruction in COPD diagnosis.

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified pulmonologist.
