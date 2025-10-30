🌿 Plant Disease Detection & Prescription

An AI-powered web application that detects plant leaf diseases using a deep learning model and provides disease details, confidence score, and health prescriptions to cure or prevent infections.

Built with TensorFlow and Streamlit, this project helps farmers, gardeners, and researchers to quickly identify plant health issues through image-based diagnosis.

🚀 Features

📸 Image Upload & Analysis – Upload a leaf image to get instant predictions.

🧠 Deep Learning Model – Uses a CNN trained on the PlantVillage dataset for high accuracy.

📊 Top-3 Prediction Visualization – Displays top disease predictions with confidence scores.

🌱 Health Diagnosis – Identifies whether the plant is healthy or infected.

💊 Prescription System – Suggests remedies, preventive tips, and best practices.

🌗 Dark Mode UI – A modern, visually appealing interface with animations and metrics.

🧩 Tech Stack

Frontend: Streamlit

Backend / Model: TensorFlow, Keras, NumPy

Data Handling: Pandas, OpenCV

Visualization: Matplotlib

Dataset: PlantVillage Dataset

⚙️ How It Works

Upload a clear image of a plant leaf.

The CNN model processes the image and predicts the disease.

The app displays:

The predicted disease name

The confidence score

Prescriptions and preventive measures

Top-3 predictions are visualized for transparency.

📂 Project Structure
plant_disease_app/
│
├── model/
│   └── plant_disease_model.h5
│
├── utils.py
├── app.py
├── requirements.txt
└── README.md

💡 Future Enhancements

📱 Mobile-friendly responsive UI

🌍 Real-time disease tracking via API integration

🗣️ Voice-based diagnosis and prescription suggestion

☁️ Deployment on Streamlit Cloud / Hugging Face Spaces
