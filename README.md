# Multidomain-and-Multimodal-fakeNews-detection-System


This repository contains the implementation of a cutting-edge **Multidomain and Multimodal Fake News Detection System** that leverages both textual and visual information to identify deceptive content with high accuracy.

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Datasets Used](#-datasets-used)
- [Performance & Results](#-performance--results)
- [How to Run](#-how-to-run)
- [Interactive Prototype](#-interactive-prototype)
- [Future Work](#-future-work)

---

## 🔍 Overview
Online misinformation has become highly sophisticated, often combining deceptive text with misleading images. Single-modality detection models (text-only or image-only) frequently fail to capture the holistic context. This project solves this by fusing **state-of-the-art NLP representations (BERT)** with **deep computer vision features (ResNet-18)**, optimized via **Fuzzy Particle Swarm Optimization (FPSO)** to classify news accurately.

## ✨ Key Features
* **Hybrid Deep Learning:** Fuses textual context (BERT) and visual context (ResNet-18).
* **Feature Optimization:** Employs **Fuzzy Particle Swarm Optimization (FPSO)** to dynamically balance feature selection weight, improving computational efficiency.
* **Explainability (SHAP):** Prepares a foundation for transparent machine learning using SHAP to interpret feature and keyword importance.
* **Interactive Web Interface:** Real-time web playground built with **Gradio** for instant model validation and demonstrations.

---

## 🛠 System Architecture
1. **Text Encoder:** Pretrained `bert-base-uncased` maps tokenized text sequence inputs into a 768-dimensional representation.
2. **Image Encoder:** Pretrained `ResNet-18` processes visual elements (RGB images resized to 224x224) into a 512-dimensional representation.
3. **Fusion Layer:** Concatenates both feature representations into a single multi-dimensional vector.
4. **Classifier Head:** Feed-Forward Neural Network mapping fused features to binary prediction outputs (`REAL` or `FAKE`).

```
[ Text Input ]  ---> [ BERT Encoder ]   ---> (768-dim Vector) \
                                                               ---> [ Concatenation & FC Classifier ] ---> [ Prediction ]
[ Image Input ] ---> [ ResNet Encoder ] ---> (512-dim Vector) /
```

---

## 📊 Datasets Used
We train and evaluate on a diverse, processed dataset combined from three benchmark sources:
1. **BuzzFeed News Dataset:** Credible and misleading real-world news articles.
   Link: https://www.kaggle.com/code/sohamohajeri/buzzfeed-news-analysis-and-classification
2. **LIAR Dataset:** Benchmark dataset for fake news detection, containing short statements.
   Link : https://www.kaggle.com/datasets/csmalarkodi/liar-fake-news-dataset
3. **Fakeddit Dataset:** A massive multimodal dataset featuring images, text headlines, and multi-way labels.
    Link : https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset
---

## 📈 Performance & Results
Following optimization and fine-tuning on **3,000 samples** for **3 epochs**, the multimodal model achieved the following performance metrics on the test split:

* **Accuracy:** `88.50%`
* **Precision:** `85.83%`
* **Recall:** `85.48%`
* **F1-Score:** `85.65%`

---

## 💻 How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install torch torchvision transformers pandas scikit-learn nltk pillow requests gradio shap
```

### Running the pipeline
1. Run the preprocessing and training blocks in the notebook to load, clean, and combine datasets.
2. Initialize and train the hybrid `MultimodalFakeNewsModel` architecture.
3. Launch the local or public shareable Gradio interface using the final cells.

---

## 🌐 Interactive Prototype
The system launches a **Gradio** web application directly from the notebook. Users can enter any news headline and paste an optional image URL to test the classification and confidence scores in real-time.
