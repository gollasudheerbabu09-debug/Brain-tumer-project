"""
Brain Tumor MRI Classifier â€” Streamlit app
Serves the fine-tuned ResNet50 (default, 93.5% test acc) and VGG16 (91.9%) models.
"""
import os

import numpy as np
import streamlit as st
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TF logs
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess  # noqa: E402

# Same order as train_generator.class_indices in both notebooks
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
IMG_SIZE = (224, 224)

# Each model must get exactly the preprocessing it was trained with
MODELS = {
    "ResNet50": {
        "path": "brain_tumor_resnet50.keras",
        "preprocess": lambda x: resnet_preprocess(x),  # caffe-style mean subtraction, BGR
    },
    "VGG16": {
        "path": "VGG16_Brain_Tumor.keras",
        "preprocess": lambda x: x / 255.0,  # rescale=1./255
    },
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="ðŸ§ ", layout="centered")


# max_entries=1: only one model in memory at a time (fits free hosting RAM limits)
@st.cache_resource(show_spinner="Loading modelâ€¦", max_entries=1)
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def prepare_image(img: Image.Image, preprocess) -> np.ndarray:
    # flow_from_directory used RGB + nearest-neighbour resize, so we match it here
    img = img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess(arr.copy())
    return np.expand_dims(arr, axis=0)


# ---------------- Sidebar ----------------
available = {name: cfg for name, cfg in MODELS.items() if os.path.exists(cfg["path"])}

with st.sidebar:
    st.header("Settings")
    if not available:
        st.error("No model files found. Put the .keras files next to app.py.")
        st.stop()
    model_name = st.selectbox("Model", list(available.keys()))
    st.markdown("---")
    st.caption(
        "Trained on the Kaggle *Brain Tumor MRI Dataset* "
        "(glioma, meningioma, pituitary, no tumor)."
    )

cfg = available[model_name]
model = load_model(cfg["path"])

# ---------------- Main ----------------
st.title("ðŸ§  Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan and the model will predict the tumor type.")

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
    except Exception:
        st.error("Could not read that file as an image.")
        st.stop()

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded scan", width="stretch")

    with st.spinner("Analyzingâ€¦"):
        probs = model.predict(prepare_image(image, cfg["preprocess"]), verbose=0)[0]

    top = int(np.argmax(probs))
    label = DISPLAY_NAMES[CLASS_NAMES[top]]
    confidence = float(probs[top])

    with col_res:
        st.subheader("Prediction")
        if CLASS_NAMES[top] == "notumor":
            st.success(f"**{label}**")
        else:
            st.warning(f"**{label}**")
        st.metric("Confidence", f"{confidence:.1%}")
        if confidence < 0.6:
            st.info("Low confidence â€” the model is unsure about this image.")

    st.subheader("Class probabilities")
    for i in np.argsort(probs)[::-1]:
        st.write(f"{DISPLAY_NAMES[CLASS_NAMES[i]]}: {probs[i]:.1%}")
        st.progress(float(probs[i]))

st.markdown("---")
st.caption(
    "âš ï¸ For research and educational purposes only. "
    "This is not a medical device and must not be used for clinical diagnosis."
)
