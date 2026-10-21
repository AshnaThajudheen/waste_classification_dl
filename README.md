# ♻️ EcoSort AI — Waste Segregation Using Deep Learning

<p align="center">
  <strong>AI-powered waste classification for a cleaner and more sustainable future.</strong>
</p>

<p align="center">
  <a href="https://wasteclassificationdl.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

## 📌 Overview

**EcoSort AI** is a deep learning-based waste classification system that identifies waste images as **Organic** or **Recyclable**. It uses a Convolutional Neural Network (CNN) trained on the Waste Classification dataset to automate image-based waste segregation.

The project provides an interactive web application built with Streamlit, allowing users to upload waste images, view predictions, explore model evaluation metrics, and learn about proper waste disposal.

## 🎯 Objectives

* Automate waste classification using deep learning.
* Distinguish organic waste from recyclable waste.
* Provide an easy-to-use image classification interface.
* Display model performance through an evaluation dashboard.
* Encourage responsible waste disposal and sustainable practices.

## ✨ Features

* **Image Classification:** Upload a waste image and classify it as Organic or Recyclable.
* **Deep Learning Model:** Uses a custom Convolutional Neural Network (CNN).
* **Confidence Scores:** Displays model scores for both categories.
* **Evaluation Dashboard:** Presents classification metrics and model performance.
* **Prediction History:** Records previous predictions in a CSV file.
* **Waste Disposal Guide:** Provides guidance on organic and recyclable waste.
* **Interactive Interface:** Built using Streamlit with a sustainability-inspired design.
* **Online Deployment:** Designed to run through Streamlit Community Cloud.

## 🧠 Model Architecture

EcoSort AI uses a custom CNN for binary image classification.

The model processes images through convolutional and pooling layers to learn visual features, followed by dense layers for classification.

**Input image size:** 224 × 224 pixels
**Input format:** RGB
**Output classes:**

* `Organic` — biodegradable waste
* `Recyclable` — materials that may be recycled

The model uses a sigmoid output for binary classification.

## 📊 Model Performance

The trained model was evaluated on the test dataset.

| Metric          |  Score |
| --------------- | -----: |
| Test Accuracy   | 86.55% |
| Macro Precision | 86.38% |
| Macro Recall    | 86.81% |
| Macro F1-Score  | 86.47% |
| ROC-AUC         | 94.08% |

### Classification Report

| Class      | Precision | Recall | F1-Score |
| ---------- | --------: | -----: | -------: |
| Organic    |    90.73% | 84.51% |   87.51% |
| Recyclable |    82.04% | 89.12% |   85.43% |

These results indicate that the model can distinguish between the two waste categories, although some images may be misclassified.

*Note: These metrics reflect the reported evaluation of the trained model. Results may vary when the model is evaluated on different datasets or real-world images.*

## 🗂️ Dataset

The project uses the **Waste Classification Data** dataset available on Kaggle.

Dataset: [Waste Classification Data — Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)

The dataset contains images organized into two categories:

| Label | Category   |
| ----- | ---------- |
| `O`   | Organic    |
| `R`   | Recyclable |

The dataset provides separate training and testing directories.

Expected directory structure:

```text
DATASET/
├── TRAIN/
│   ├── O/
│   └── R/
└── TEST/
    ├── O/
    └── R/
```

## 🛠️ Technologies Used

| Technology                | Purpose                              |
| ------------------------- | ------------------------------------ |
| Python                    | Core programming language            |
| TensorFlow / Keras        | Deep learning model development      |
| CNN                       | Image classification                 |
| NumPy                     | Numerical operations                 |
| Pandas                    | Prediction history and data handling |
| Pillow                    | Image processing                     |
| Streamlit                 | Interactive web application          |
| GitHub                    | Version control and project hosting  |
| Streamlit Community Cloud | Web application deployment           |

## 📁 Project Structure

```text
waste_classification_dl/
├── app.py
├── requirements.txt
├── README.md
└── ecosort_best.keras
```

* `app.py` — Streamlit application.
* `requirements.txt` — Python dependencies.
* `README.md` — Project documentation.
* `ecosort_best.keras` — Trained CNN model.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AshnaThajudheen/waste_classification_dl.git
```

### 2. Navigate to the Project Directory

```bash
cd waste_classification_dl
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Verify the Project Files

Ensure that `app.py` and `ecosort_best.keras` are in the same directory.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🌐 Live Demo

**Try EcoSort AI:** https://wasteclassificationdl.streamlit.app/

If the application is temporarily unavailable, check the deployment status or run it locally using the installation instructions above.

## 🔮 Future Enhancements

* Expand classification to additional waste categories.
* Improve model accuracy and generalization.
* Explore transfer learning with pretrained CNN architectures.
* Optimize inference speed for mobile and edge devices.
* Integrate real-time camera-based waste detection.
* Add persistent cloud storage for prediction history.
* Develop a waste segregation recommendation system.

## 🌱 Impact

EcoSort AI demonstrates how deep learning can support waste management by making image-based waste classification more accessible. The system aims to encourage better segregation practices and contribute to environmental awareness.

## 👩‍💻 Author

**Ashna Thajudheen**

B.Tech — Computer Science and Engineering (Artificial Intelligence)

GitHub: [@AshnaThajudheen](https://github.com/AshnaThajudheen)
