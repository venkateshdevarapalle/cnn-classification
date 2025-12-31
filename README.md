# 🐶🐱 Intelligent Dog–Cat Image Classifier (Streamlit Web App)

An end-to-end **Deep Learning–based image classification system** that accurately identifies whether an uploaded image contains a **Dog** or a **Cat**.  
If the uploaded image does **not** belong to either category, the system intelligently rejects it using **confidence-based unknown image detection**.

The application is deployed as an **interactive Streamlit web application**.

---

## 🚀 Features

- ✅ Dog vs Cat image classification  
- ❌ Unknown image rejection (confidence thresholding)  
- 🧠 Transfer Learning using **MobileNetV2**  
- 🌐 Interactive **Streamlit Web Interface**  
- ⚡ Fast training configuration (CPU-friendly)  
- 📊 Displays prediction confidence  

---

## 🧠 Model Overview

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Approach:** Transfer Learning
- **Input Size:** 160 × 160 RGB images
- **Output Classes:**  
  - Cat  
  - Dog  
- **Unknown Detection:**  
  - If prediction confidence < **0.60**, output:  
    > *Neither Cat nor Dog – Please upload a valid image.*

---

## 🗂️ Project Structure

CNN-Classification/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── predict.py # Prediction & confidence logic
├── data_loader.py # Dataset loader (auto-detect folders)
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore


---

## 📂 Dataset

- **Dataset Name:** Dog and Cat Classification Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset  

> ⚠️ Dataset is **not included** in this repository due to size constraints.

Expected dataset structure:

PetImages/
├── Cat/
└── Dog/


---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Model Architecture:** MobileNetV2  
- **Web Framework:** Streamlit  
- **Libraries:** NumPy, Pandas, Pillow, OpenCV, SciPy  

---

##  How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Venkateshdevarapalle/CNN-Classification.git
cd CNN-Classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train_model.py


This will generate:

dog_cat_model.h5

4️⃣ Run the Web Application
python -m streamlit run app.py


Open in browser:

http://localhost:8501

