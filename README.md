# 🤖 SETI Signal Classifier

Image classifier built with Convolutional Neural Networks (CNN) and Class Activation Map (CAM) support — a project for classifying radio-signal-style images and visualizing model interpretation during training.

Live demo: https://huggingface.co/spaces/Esabelle/seti_classifier_demo

training notebook: 
https://www.kaggle.com/code/esabellechen/seti-signal-classifier-vgg16-pytorch
https://www.kaggle.com/code/esabellechen/seti-signal-classifier-googlene-pytorch

---

## 🔍 Overview

This project implements a **deep learning-based image classification pipeline** using CNNs to classify images and interpret model decisions through **Class Activation Maps (CAMs)**.

It is especially suited for visualizing how the network focuses on areas of the image during training and inference. The notebook and code include:

✔️ CNN model training  
✔️ Image preprocessing  
✔️ CAM visualization  
✔️ Evaluation and plotting

## Demo Video
[![Watch Demo](https://img.youtube.com/vi/yxtqx7oFAUk/maxresdefault.jpg)](https://youtu.be/yxtqx7oFAUk)

https://youtu.be/yxtqx7oFAUk
<!-- <img width="995" height="770" alt="Screenshot 2026-04-07 at 8 11 36 PM" src="https://github.com/user-attachments/assets/7b19b3c5-2a4a-4605-9e89-a1bf91fe70f0" /> -->

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

## Prerequisites

- Python 3.11
- Docker & Docker Compose (optional)
- 4GB+ RAM (for transformer models)

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
    git clone https://github.com/Esabelle11/seti_classifier.git
    cd seti_classifier
    ```


2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## Docker Setup

### Using Docker Compose (Recommended)

1. **Build and run**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Frontend: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

3. **Stop the application**
   ```bash
   docker-compose down
   ```

### Using Docker Only

1. **Build the image**
   ```bash
   docker build -t seti-classifier .
   ```

2. **Run the container**
   ```bash
   docker run -p 8888:8888 seti-classifier
   ```
