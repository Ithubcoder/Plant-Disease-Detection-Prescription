import numpy as np
from PIL import Image
import json, os

def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_class_names():
    json_path = "model/class_indices.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError("❌ class_indices.json not found. Train the model first.")
    with open(json_path, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # reverse mapping
