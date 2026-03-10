# 🌿 Crop Disease Detection System
ML + Deep Learning pipeline for plant disease classification using leaf images | PlantVillage | HOG → XGBoost → EfficientNet

> A progressive ML + Deep Learning pipeline for detecting plant leaf diseases from images.  
> Trained on the PlantVillage dataset — 38 disease classes across 14 crop species.

---

## 📌 Overview

This project builds a crop disease detection system in **three progressive stages**:

| Stage | Approach | Expected Accuracy |
|-------|----------|-------------------|
| 1 | HOG Features + Random Forest / XGBoost | ~80–88% |
| 2 | Pretrained CNN Embeddings + XGBoost | ~90–93% |
| 3 | Fine-tuned EfficientNet (End-to-End DL) | ~95%+ |

Each stage builds on the previous, allowing clear comparison between classical ML and deep learning approaches.

---

## 📁 Project Structure

```
crop-disease-detection/
│
├── data/
│   └── plantvillage/
│       └── color/                  # 38 class folders (folder name = label)
│
├── stage1_ml/
│   ├── extract_features.py         # HOG + color histogram extraction
│   ├── train.py                    # Random Forest / XGBoost training
│   └── evaluate.py                 # Accuracy, confusion matrix
│
├── stage2_embeddings/
│   ├── extract_embeddings.py       # Pretrained EfficientNet embeddings
│   └── train_xgboost.py            # XGBoost on embeddings
│
├── stage3_deeplearning/
│   ├── dataset.py                  # PyTorch Dataset + DataLoader
│   ├── model.py                    # EfficientNet fine-tuning
│   ├── train.py                    # Training loop
│   └── evaluate.py                 # Metrics + Grad-CAM
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_stage1_baseline.ipynb
│   ├── 03_stage2_embeddings.ipynb
│   └── 04_stage3_deeplearning.ipynb
│
├── app/
│   └── app.py                      # Streamlit inference UI
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**PlantVillage Dataset**
- 54,306 labeled leaf images
- 14 crop species, 26 diseases, 38 total classes
- Labels encoded as folder names (e.g. `Tomato___Early_blight`)

**Download:**
- Kaggle: [plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- HuggingFace: `load_dataset("mohanty/PlantVillage", "color")`

After downloading, extract and point `data_dir` to the `color/` subfolder.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/crop-disease-detection.git
cd crop-disease-detection
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
pandas
scikit-learn
xgboost
opencv-python
scikit-image
matplotlib
seaborn
torch
torchvision
efficientnet-pytorch
streamlit
```

---

## 🚀 Usage

### Stage 1 — Classical ML Baseline

```bash
# Extract HOG features and train XGBoost
python stage1_ml/extract_features.py --data_dir data/plantvillage/color/
python stage1_ml/train.py
python stage1_ml/evaluate.py
```

### Stage 2 — CNN Embeddings + XGBoost

```bash
python stage2_embeddings/extract_embeddings.py --data_dir data/plantvillage/color/
python stage2_embeddings/train_xgboost.py
```

### Stage 3 — Fine-tuned EfficientNet

```bash
python stage3_deeplearning/train.py --epochs 20 --batch_size 32 --lr 0.001
python stage3_deeplearning/evaluate.py
```

### Run Streamlit App

```bash
streamlit run app/app.py
```

Upload a leaf image → get predicted disease + confidence score.

---

## 📊 Results

| Stage | Model | Accuracy | F1 Score |
|-------|-------|----------|----------|
| 1 | XGBoost + HOG | TBD | TBD |
| 2 | XGBoost + EfficientNet Embeddings | TBD | TBD |
| 3 | Fine-tuned EfficientNet | TBD | TBD |

> Results will be updated after training runs.

---

## 🗺️ Roadmap

- [x] Dataset acquisition and EDA
- [ ] Stage 1: HOG + XGBoost baseline
- [ ] Stage 2: CNN embeddings + XGBoost
- [ ] Stage 3: EfficientNet fine-tuning
- [ ] Streamlit deployment UI
- [ ] Treatment recommendation module (RAG-based)
- [ ] Mobile-friendly export (ONNX / TFLite)

---

## 🧠 Key Concepts Used

- **HOG (Histogram of Oriented Gradients)** — handcrafted image features for classical ML
- **Transfer Learning** — leveraging pretrained ImageNet weights
- **Fine-tuning** — adapting pretrained CNN to domain-specific data
- **Grad-CAM** — visualizing what the model focuses on in a leaf image

---

## 👤 Author

**Apoorva**  
Aspiring Data Scientist | ML • Deep Learning • Generative AI  
[GitHub](#) • [LinkedIn](#)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
