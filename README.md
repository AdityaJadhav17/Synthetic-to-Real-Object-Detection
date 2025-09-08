# Synthetic-to-Real Object Detection Challenge

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/synthetic-2-real-object-detection-challenge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Python](https://img.shields.io/badge/Python-3.10+-green)

## 📖 Overview
This repository contains my solution pipeline for the **Synthetic-to-Real Object Detection Challenge** on Kaggle.  
The task: **train object detectors on synthetic images of a cheerio box** and generalize to **real test images**.  
The challenge highlights domain adaptation and sim-to-real transfer in computer vision.

---

## 📂 Repo Layout
```
synthetic-to-real-detection/
├─ data/                                         # training & test datasets
├─ ENV_SETUP/                                    # environment setup files (requirements, setup scripts)
├─ predictions/                                  # generated prediction CSVs & bounding box files
├─ runs/                                         # YOLOv8 training outputs (weights, logs, metrics)
├─ classes                                       # text file with class labels
├─ convert_preds_to_csv.py                       # converts YOLO predictions to Kaggle submission format
├─ predict.py                                    # run inference on test images
├─ submission.xlsx                               # my Kaggle submission
├─ train.py                                      # training script for YOLOv8 models
├─ yolo_params.yaml                              # config file with YOLO hyperparameters
└─ yolov8s.pt                                    # pretrained YOLOv8 model weights

```

---

## ⚙️ Setup

### 1️⃣ Create Environment
```bash
# clone repo
git clone https://github.com/your-username/synthetic-to-real-detection.git
cd synthetic-to-real-detection

# create environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# install dependencies
pip install -r ENV_SETUP/requirements.txt
```
2️⃣ Download Dataset
Download from the Kaggle competition:

Synthetic-to-Real Object Detection Challenge

Place under data/:

```bash

data/
├─ train/      # synthetic images + labels
├─ test/       # real images
└─ sample/     # sample_submission.csv
```
---

🚀 Usage

Train YOLOv8
```bash
python train.py --config yolo_params.yaml
Run Inference

python predict.py --weights runs/train/exp/weights/best.pt --source data/test/
Convert Predictions to Submission

python convert_preds_to_csv.py --predictions predictions/ --output submission.csv
Upload submission.csv to Kaggle for evaluation.
```
---

🧪 Experiments

Baseline: YOLOv8s pretrained on COCO → fine-tuned on synthetic dataset

Augmentations: applied color jitter, blur, random rotations

Domain Randomization: experimented with synthetic filtering & stronger augmentations

Model Variants: tested YOLOv8n/s/m with different hyperparameters

---

📊 Results

Final Kaggle leaderboard scores:

Dataset Split	mAP Score
Public LB	0.91750
Private LB	0.90738

🏆 Achieved Top-tier performance (>0.90 mAP) on both public and private leaderboards, showing strong generalization across the sim-to-real domain gap.

---

🔑 Key Learnings

Synthetic data is powerful, but augmentation and domain randomization are critical to bridge the sim-to-real gap.

Smaller models (YOLOv8n/s) train faster but may underfit; larger backbones (YOLOv8m) improve generalization.

Pretraining on large datasets like COCO followed by fine-tuning on synthetic data accelerates convergence.

Data quality matters: filtering out unrealistic synthetic samples improved robustness.

---

📜 License
This project is licensed under the MIT License.

---

🙌 Acknowledgments

Ultralytics YOLOv8

Kaggle Competition

Duality AI for organizing and hosting the dataset

---
