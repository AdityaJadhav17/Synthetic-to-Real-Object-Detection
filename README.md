# Synthetic to Real Object Detection

## 🔍 Overview

This repository documents my submission for the [Synthetic to Real Object Detection Challenge](https://www.kaggle.com/competitions/synthetic-2-real-object-detection-challenge/overview) hosted by **Duality AI** on **April 4th, 2025**. 

The competition addressed a key challenge in computer vision: building object detection models trained exclusively on **synthetic images** that generalize effectively to **real-world data**. As members of our college AI Club, we collaborated to create a high-performing model that achieved **92% accuracy** and placed in the **Top 10** on the leaderboard. 

This challenge emphasized the growing role of synthetic data in scalable, transferable AI systems and gave us hands-on experience in sim-to-real model development.

## 🛠️ Approach & Methodology

We focused on designing a robust training pipeline capable of handling domain shifts between synthetic and real data. Key components of our approach included:

- 🔁 **Domain Randomization**: Applied randomized textures, lighting, and object variations during training to improve generalization.
- 🧪 **Data Augmentation**: Used geometric and photometric augmentations (rotation, brightness, noise) to diversify the training set.
- 🧠 **YOLOv8 Backbone**: Chose a high-performance, real-time object detection architecture for its speed and accuracy.
- 🔄 **Transfer Learning**: Fine-tuned a pre-trained model on the synthetic dataset to leverage existing knowledge.
- ⚙️ **Custom Training Pipeline**: Built a modular pipeline for efficient experimentation and tracking of hyperparameter performance.

These strategies helped us overcome the sim-to-real gap and build a detector that performed reliably on real-world test images.
