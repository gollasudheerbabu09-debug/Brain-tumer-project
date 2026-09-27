"""
Brain Tumor MRI Classifier — Streamlit app
Serves a fine-tuned ViT-B/16 (default, exported from PyTorch to ONNX), ResNet50,
VGG16, and a Vision Transformer built from scratch.
"""
import os

import numpy as np
import streamlit as st
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TF logs
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, Model  # noqa: E402
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess  # noqa: E402
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess  # noqa: E402

# Same order as train_generator.class_indices in both notebooks
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
IMG_SIZE = (224, 224)


# ---------------- Vision Transformer (built from scratch) ----------------
# The custom layers below must match the notebook exactly so the saved
# weights line up. The architecture is rebuilt here and only the weights are
# loaded, because the custom layers were saved without get_config().
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)


def build_vit(num_classes=4, image_size=224, patch_size=16, projection_dim=64,
              num_heads=4, transformer_layers=4, dropout_rate=0.2):
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim]

    inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim,
                                         dropout=dropout_rate)(x1, x1)
        x2 = layers.Add()([attn, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(transformer_units[1])(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        encoded = layers.Add()([x3, x2])

    rep = layers.LayerNormalization(epsilon=1e-6)(encoded)
    rep = layers.Flatten()(rep)
    rep = layers.Dropout(0.5)(rep)
    rep = layers.Dense(256, activation="relu")(rep)
    rep = layers.Dropout(0.5)(rep)
    outputs = layers.Dense(num_classes, activation="softmax")(rep)
    return Model(inputs=inputs, outputs=outputs)


def load_keras_model(path):
    return tf.keras.models.load_model(path, compile=False)


def load_vit(path):
    model = build_vit()
    model.load_weights(path)
    return model


# ---------------- ViT-B/16 (PyTorch -> ONNX) ----------------
# Normalization stats of timm vit_base_patch16_224.augreg2_in21k_ft_in1k
VIT_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
VIT_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class OnnxViT:
    """Makes the ONNX model look like a Keras model: predict() returns probabilities."""

    def __init__(self, path):
        import onnxruntime as ort  # imported lazily so other models don't need it

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x, verbose=0):
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)  # NHWC -> NCHW for PyTorch
        logits = self.session.run(None, {self.input_name: x})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def load_onnx_vit(path):
    return OnnxViT(path)


# Each model must get exactly the preprocessing it was trained with
MODELS = {
    "ViT-B/16 (pretrained)": {
        "path": "vit_b16.onnx",
        "loader": load_onnx_vit,
        "preprocess": lambda x: (x / 255.0 - VIT_MEAN) / VIT_STD,
        "resample": Image.BILINEAR,  # torchvision Resize uses bilinear
    },
    "ResNet50": {
        "path": "brain_tumor_resnet50.keras",
        "loader": load_keras_model,
        "preprocess": lambda x: resnet_preprocess(x),  # caffe-style mean subtraction, BGR
    },
    "VGG16": {
        "path": "VGG16_Brain_Tumor_v2.keras",
        "loader": load_keras_model,
        "preprocess": lambda x: vgg_preprocess(x),  # vgg16.preprocess_input (caffe-style)
        "resize": "tf_bilinear",  # image_dataset_from_directory resizes with tf.image.resize
    },
    "Vision Transformer (from scratch)": {
        "path": "best_vit_model.keras",
        "loader": load_vit,
        "preprocess": lambda x: x / 255.0,  # rescale=1./255
    },
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


# max_entries=1: only one model in memory at a time (fits free hosting RAM limits)
@st.cache_resource(show_spinner="Loading model…", max_entries=1)
def load_model(name: str):
    cfg = MODELS[name]
    return cfg["loader"](cfg["path"])


def prepare_image(img: Image.Image, cfg: dict) -> np.ndarray:
    img = img.convert("RGB")
    if cfg.get("resize") == "tf_bilinear":
        # Same resize as keras.utils.image_dataset_from_directory
        arr = tf.image.resize(np.asarray(img, dtype=np.float32), IMG_SIZE, method="bilinear").numpy()
    else:
        # Keras flow_from_directory used nearest-neighbour; torchvision used PIL bilinear
        img = img.resize(IMG_SIZE, cfg.get("resample", Image.NEAREST))
        arr = np.asarray(img, dtype=np.float32)
    arr = cfg["preprocess"](arr.copy())
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
model = load_model(model_name)

# ---------------- Main ----------------
st.title("🧠 Brain Tumor MRI Classifier")
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

    with st.spinner("Analyzing…"):
        probs = model.predict(
            prepare_image(image, cfg), verbose=0
        )[0]

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
            st.info("Low confidence — the model is unsure about this image.")

    st.subheader("Class probabilities")
    for i in np.argsort(probs)[::-1]:
        st.write(f"{DISPLAY_NAMES[CLASS_NAMES[i]]}: {probs[i]:.1%}")
        st.progress(float(probs[i]))

st.markdown("---")
st.caption(
    "⚠️ For research and educational purposes only. "
    "This is not a medical device and must not be used for clinical diagnosis."
)
