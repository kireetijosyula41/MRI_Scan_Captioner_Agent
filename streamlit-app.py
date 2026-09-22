import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import openai

model = load_model('brain_tumor.h5')

api_key = st.secrets["OPENAI_API_KEY"]

# Loading of the image for the streamlit app
def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Return the softmax distribution and its predicted class details
def predict_image(image_array):
    distribution = model.predict(image_array)[0]
    predicted_class = int(np.argmax(distribution))
    labels = {0: "Glioma", 1: "Meningioma", 2: "No tumor", 3: "Pituitary"}
    predicted_label = labels[predicted_class]
    confidence = float(distribution[predicted_class])
    class_probabilities = {
        label: float(distribution[index]) for index, label in labels.items()
    }
    return distribution, predicted_label, confidence, class_probabilities


def get_confidence_tier(confidence):
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "moderate"
    return "low"


client = openai.OpenAI(api_key=api_key)

# Image caption function using gpt-4o-mini
def generate_caption(predicted_label, confidence_tier):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Predicted class: {predicted_label}. Confidence tier: {confidence_tier}. Write a one sentence description for this MRI scan showing the predicted class. Indicate the type of tumor if present, the location in the brain where the tumor is, and the view of the brain the user is looking at."}
        ]
    )
    return response.choices[0].message.content

# Streamlit app construction.
st.title('Brain Tumor Detection App')
uploaded_file = st.file_uploader("Upload an MRI scan", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", use_container_width=True)
    distribution, predicted_label, confidence, class_probabilities = predict_image(image)
    confidence_tier = get_confidence_tier(confidence)
    st.write(f"Predicted class: {predicted_label}")
    st.write(f"Confidence: {confidence:.2%}")
    st.write(f"Confidence tier: {confidence_tier.capitalize()}")
    if confidence_tier == "low":
        st.warning("Low confidence: the model's prediction is uncertain.")
    st.bar_chart(
        pd.DataFrame.from_dict(
            class_probabilities, orient="index", columns=["Probability"]
        )
    )
    caption = generate_caption(predicted_label, confidence_tier)
    st.write(f"AI Generated Caption from OPENAI: {caption}")
