import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import openai

model = load_model('brain_tumor.h5')

api_key = st.secrets["OPENAI_API_KEY"]

def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_array):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    labels = {0: "Glioma", 1: "Meningioma", 2: "No tumor", 3: "Pituitary"}
    return labels[predicted_class]

client = openai.OpenAI(api_key=api_key)

def generate_caption(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Write a one sentence description for this MRI scan showing the presence of {prompt}. Indicate the type of tumor if present, the location in the brain where the tumor is, and the view of the brain the user is looking at"}
        ]
    )
    return response.choices[0].message.content

st.title('Brain Tumor Detection App')
uploaded_file = st.file_uploader("Upload an MRI scan", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", use_container_width=True)
    description = predict_image(image)
    caption = generate_caption(description)
    st.write(f"AI Generated Caption from OPENAI: {caption}")