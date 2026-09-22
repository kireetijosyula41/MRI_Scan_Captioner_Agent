import json
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import colormaps
import openai
from torch_classifier import grad_cam_heatmap, load_classifier, predict_probabilities


@st.cache_resource
def get_model():
    return load_classifier('brain_tumor.h5')


model = get_model()

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
    distribution = predict_probabilities(model, image_array)
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


def create_grad_cam_overlay(image_array, original_image, class_index):
    heatmap = grad_cam_heatmap(
        model, image_array, class_index,
        (original_image.height, original_image.width),
    )

    colors = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    color_image = Image.fromarray(colors)
    opacity = Image.fromarray((heatmap * 128).astype(np.uint8))
    return Image.composite(color_image, original_image.convert("RGB"), opacity)


client = openai.OpenAI(api_key=api_key)

# Generate a report from the classifier output, without inferring image findings
def generate_report(prediction):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Write an informational report using only the supplied classifier "
                    "prediction data. Do not invent image findings. Do not assert or "
                    "imply tumor location, size, imaging view, clinical diagnosis, "
                    "prognosis, or treatment unless directly supported by the supplied "
                    "data. A predicted class and its probabilities do not establish "
                    "those facts. Clearly distinguish the model prediction from a "
                    "clinical diagnosis. Output exactly these five Markdown sections "
                    "in order: ## Model prediction, ## Confidence, "
                    "## Interpretation, ## Limitations, ## Safety note. "
                    "Report the supplied label, confidence, confidence tier, and "
                    "limitations accurately. In the safety note, state that the "
                    "report is not a medical diagnosis and should be reviewed by "
                    "a qualified clinician."
                ),
            },
            {
                "role": "user",
                "content": f"Classifier prediction data:\n{json.dumps(prediction, indent=2)}",
            },
        ]
    )
    return response.choices[0].message.content

# Streamlit app construction.
st.title('Brain Tumor Detection App')
uploaded_file = st.file_uploader("Upload an MRI scan", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    uploaded_file.seek(0)
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Uploaded MRI scan", use_container_width=True)
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
    overlay = create_grad_cam_overlay(
        image, original_image, int(np.argmax(distribution))
    )
    st.image(overlay, caption=f"Grad-CAM for {predicted_label}", use_container_width=True)
    st.caption(
        "Grad-CAM is a model-attention visualization, not tumor segmentation. "
        "Highlighted regions show what influenced this prediction."
    )
    prediction = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "probabilities": class_probabilities,
        "limitations": (
            "The classifier provides class probabilities only. It does not establish "
            "a clinical diagnosis or determine tumor location, size, imaging view, "
            "prognosis, or treatment."
        ),
    }
    report = generate_report(prediction)
    st.markdown(report)
