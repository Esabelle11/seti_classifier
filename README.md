# 🤖 SETI Signal Classifier

Image classifier built with Convolutional Neural Networks (CNN) and Class Activation Map (CAM) support — a project for classifying radio-signal-style images and visualizing model interpretation during training.

Live demo: https://huggingface.co/spaces/Esabelle/seti_classifier_demo

---

## 🔍 Overview

This project implements a **deep learning-based image classification pipeline** using CNNs to classify images and interpret model decisions through **Class Activation Maps (CAMs)**.

It is especially suited for visualizing how the network focuses on areas of the image during training and inference. The notebook and code include:

✔️ CNN model training  
✔️ Image preprocessing  
✔️ CAM visualization  
✔️ Evaluation and plotting

---

## 📂 Project Contents

| Folder / File | Description |
|---------------|-------------|
| `app/` | Source for model training and classification logic |
| `images/test/` | Example test images |
| `model/` | Saved model checkpoints |
| `seti-image-classification-notebook.ipynb` | Notebook for exploration & training |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | For containerized setup |
| `docker-compose.yml` | For running with Docker services |

---

## 🛠️ Features

- 📷 **Image Classification** using Convolutional Neural Networks  
- 🧠 **Class Activation Maps (CAM)** for visual interpretability  
- 📊 Training + evaluation workflow  
- 🧪 Test images included  
- 🐳 Docker support for repeatable environment

---

## 🚀 Get Started

### 1. Clone the repository

```bash
git clone https://github.com/Esabelle11/seti_classifier.git
cd seti_classifier