import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import streamlit as st
import torch
import timm
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

# Model and class config
MODEL_PATH = "efficientnetv2s_best.pth"
IMG_SIZE = 224
CLASSES_TXT = "dataset/classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
def load_classes(classes_txt):
    import pandas as pd
    classes = pd.read_csv(classes_txt, sep=" ", header=None, names=["label", "name"])
    return classes["name"].tolist()

# Load model
def load_model(model_path, num_classes):
    model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# Preprocess image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = torch.tensor(image).unsqueeze(0).to(DEVICE)
    return image

# Streamlit UI
st.title("Insect Classifier Test App")

classes = load_classes(CLASSES_TXT)
model = load_model(MODEL_PATH, len(classes))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred = np.argmax(probabilities)
        confidence = probabilities[pred] * 100
        st.write(f"Prediction: {pred} - {classes[pred]}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write("Class probabilities:")
        prob_table = []
        for idx, (cls, prob) in enumerate(zip(classes, probabilities)):
            prob_table.append({"Class #": idx, "Class Name": cls, "Confidence (%)": f"{prob*100:.2f}"})
        st.table(prob_table)
