# 🌿 Plant Disease Detection & Prescription

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Project Overview

**Plant Disease Detection & Prescription** is an AI-powered web application that detects plant leaf diseases using a deep learning model and provides **disease information, confidence score, and cure suggestions**.  
The app is built using **TensorFlow** and **Streamlit** for real-time diagnosis and user interaction.

---

## 🚀 Features

- 📸 Upload leaf images for instant disease prediction  
- 🧠 Deep Learning model trained on the *PlantVillage* dataset  
- 📊 Displays top-3 prediction results with confidence levels  
- 🌱 Classifies plant health (Healthy or Infected)  
- 💊 Suggests remedies and preventive measures  
- 🌗 Modern UI with metrics and visuals  

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|------------------|
| **Frontend** | Streamlit |
| **Backend / Model** | TensorFlow, Keras, NumPy |
| **Data Handling** | Pandas, OpenCV |
| **Visualization** | Matplotlib |
| **Dataset** | [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) |

---

## ⚙️ How It Works

1. Upload a clear image of the plant leaf.  
2. The CNN model processes the image.  
3. The system predicts:
   - 🦠 Disease name  
   - 📈 Confidence score  
   - 💊 Prescription and prevention methods  
4. Visualizes top-3 predictions using a bar graph.

---

## 📂 Project Structure

plant_disease_app/
│
├── model/
│ └── plant_disease_model.h5
│
├── app.py
├── utils.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🧪 Installation & Usage

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
streamlit run app.py
Upload a plant leaf image and get instant disease detection with prescription!

💡 Future Enhancements
📱 Mobile-optimized responsive UI

🌍 Real-time API for crop health monitoring

🗣️ Voice-based diagnosis system

☁️ Deploy on Streamlit Cloud / Hugging Face Spaces
