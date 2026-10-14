"""
Brain Tumor MRI Classifier — Streamlit app
Serves a fine-tuned ViT-B/16 (default, exported from PyTorch to ONNX), ResNet50,
VGG16, and a Vision Transformer built from scratch.
"""
import base64
import html
import io
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
    "ViT-B/16": {
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

st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠", layout="wide")

SAMPLE_DIR = "samples"
CLASS_COLORS = {
    "glioma": "#B04A3A",
    "meningioma": "#B9862A",
    "notumor": "#3D8669",
    "pituitary": "#6656A0",
}


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


def list_samples():
    """Sample scans named <class>_<n>.jpg, e.g. glioma_1.jpg."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    samples = []
    for f in files:
        cls = f.split("_")[0].lower()
        samples.append({"path": os.path.join(SAMPLE_DIR, f), "true": cls if cls in CLASS_NAMES else None})
    return samples


def to_data_uri(img: Image.Image, size=560) -> str:
    img = img.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------- Styles ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input { font-family: 'Public Sans', system-ui, sans-serif; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
h1, h2, h3 { font-family: 'Public Sans', system-ui, sans-serif; letter-spacing: -0.01em; }

.page-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0 0 .35rem 0; }
.page-lede { font-size: 1.02rem; opacity: .75; max-width: 62ch; margin: 0 0 1.4rem 0; line-height: 1.55; }

/* The light box: the scan shown the way it would be on a reading-room monitor */
.lightbox {
  position: relative; background: #0A1014; border-radius: 6px; overflow: hidden;
  aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center;
  box-shadow: inset 0 0 0 1px #24323B;
}
.lightbox img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
.overlay {
  position: absolute; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .78rem;
  color: #D7E3EA; line-height: 1.45; text-shadow: 0 1px 2px rgba(0,0,0,.9); pointer-events: none;
}
.overlay.tl { top: 12px; left: 14px; }
.overlay.tr { top: 12px; right: 14px; text-align: right; }
.overlay.bl { bottom: 12px; left: 14px; font-size: .95rem; color: #FFFFFF; }
.overlay .dot { display: inline-block; width: .6em; height: .6em; border-radius: 50%; margin-right: .45em; vertical-align: .05em; }

/* Result panel */
.verdict-label { font-size: .9rem; opacity: .65; margin-bottom: .1rem; }
.verdict { font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin-bottom: .2rem; }
.verdict-conf { font-size: 1.05rem; opacity: .8; margin-bottom: 1.1rem; }
.truth { font-size: .92rem; padding: .55rem .75rem; border-radius: 6px; margin-bottom: 1.1rem; }
.truth.ok { background: rgba(61,134,105,.12); color: #2C6B52; }
.truth.miss { background: rgba(176,74,58,.12); color: #8E3A2D; }
.prob-row { display: grid; grid-template-columns: 9.5rem 1fr 3.6rem; align-items: center; gap: .75rem; margin: .55rem 0; font-size: .95rem; }
.prob-track { height: 10px; background: rgba(127,140,150,.18); border-radius: 5px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 5px; }
.prob-pct { text-align: right; font-variant-numeric: tabular-nums; opacity: .85; }
.prob-row.top { font-weight: 600; }

.section-title { font-size: 1.25rem; font-weight: 650; margin: 2.2rem 0 .25rem 0; }
.section-note { font-size: .93rem; opacity: .7; margin: 0 0 1rem 0; }
.sample-caption { font-size: .85rem; opacity: .75; margin: .25rem 0 .35rem 0; }
.footer-note { font-size: .82rem; opacity: .6; margin-top: 2.5rem; max-width: 70ch; }

@media (max-width: 640px) {
  .page-title { font-size: 1.6rem; }
  .verdict { font-size: 1.8rem; }
  .prob-row { grid-template-columns: 7.5rem 1fr 3.2rem; font-size: .88rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Models ----------------
available = {name: cfg for name, cfg in MODELS.items() if os.path.exists(cfg["path"])}
if not available:
    st.error("No model files found. Put the model files next to app.py, then restart the app.")
    st.stop()

samples = list_samples()

# Which scan is being shown: a sample (by path) or the uploaded file
if "source" not in st.session_state:
    st.session_state.source = ("sample", samples[0]["path"]) if samples else None


def use_upload():
    if st.session_state.get("uploader") is not None:
        st.session_state.source = ("upload", None)


def use_sample(path):
    st.session_state.source = ("sample", path)


# ---------------- Header ----------------
st.markdown('<div class="page-title">Brain MRI tumor classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-lede">Pick a sample scan below or upload your own. The model sorts it into '
    "glioma, meningioma, pituitary tumor, or no tumor, and shows how confident it is in each.</p>",
    unsafe_allow_html=True,
)

model_name = st.radio("Model", list(available.keys()), horizontal=True)
cfg = available[model_name]
model = load_model(model_name)

# ---------------- Current scan ----------------
image, true_cls, source_text = None, None, ""
src = st.session_state.source
if src and src[0] == "upload" and st.session_state.get("uploader") is not None:
    try:
        image = Image.open(st.session_state.uploader)
        source_text = "Your upload"
    except Exception:
        st.error("That file isn't a readable image. Upload a JPG or PNG scan.")
elif samples:
    # A sample is selected, or an upload was removed: show a sample
    path = src[1] if src and src[0] == "sample" and os.path.exists(src[1]) else samples[0]["path"]
    st.session_state.source = ("sample", path)
    src = st.session_state.source
    image = Image.open(path)
    true_cls = next((s["true"] for s in samples if s["path"] == path), None)
    source_text = "Sample scan"

if image is None:
    st.info("Upload an MRI scan below to see a prediction.")
else:
    with st.spinner("Reading the scan…"):
        probs = model.predict(prepare_image(image, cfg), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = CLASS_NAMES[int(order[0])]
    conf = float(probs[order[0]])

    col_scan, col_result = st.columns([1.05, 1], gap="large")

    with col_scan:
        st.markdown(
            f"""
<div class="lightbox">
  <img src="{to_data_uri(image)}" alt="MRI scan being classified">
  <div class="overlay tl">{html.escape(source_text)}</div>
  <div class="overlay tr">{html.escape(model_name)}</div>
  <div class="overlay bl"><span class="dot" style="background:{CLASS_COLORS[top]}"></span>{DISPLAY_NAMES[top]} {conf:.1%}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_result:
        st.markdown('<div class="verdict-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict" style="color:{CLASS_COLORS[top]}">{DISPLAY_NAMES[top]}</div>'
            f'<div class="verdict-conf">{conf:.1%} confidence</div>',
            unsafe_allow_html=True,
        )
        if true_cls:
            if true_cls == top:
                st.markdown(
                    f'<div class="truth ok">Correct. This sample is labelled {DISPLAY_NAMES[true_cls]}.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="truth miss">Missed. This sample is labelled {DISPLAY_NAMES[true_cls]}.</div>',
                    unsafe_allow_html=True,
                )
        rows = []
        for i in order:
            c = CLASS_NAMES[int(i)]
            p = float(probs[i])
            rows.append(
                f'<div class="prob-row{" top" if c == top else ""}">'
                f"<span>{DISPLAY_NAMES[c]}</span>"
                f'<div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;background:{CLASS_COLORS[c]}"></div></div>'
                f'<span class="prob-pct">{p:.1%}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        if conf < 0.6:
            st.markdown(
                '<p class="section-note" style="margin-top:1rem">The model is unsure about this scan. '
                "Try another model or a clearer image.</p>",
                unsafe_allow_html=True,
            )

# ---------------- Upload ----------------
st.markdown('<div class="section-title">Upload your own scan</div>', unsafe_allow_html=True)
st.file_uploader(
    "Drag a JPG or PNG here from your computer, or click to browse",
    type=["jpg", "jpeg", "png"],
    key="uploader",
    on_change=use_upload,
)

# ---------------- Sample gallery ----------------
if samples:
    st.markdown('<div class="section-title">Try a sample scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">These come from the held-out test set, so the models never saw them '
        "during training. Each one is labelled with its true class.</p>",
        unsafe_allow_html=True,
    )
    current = src[1] if src and src[0] == "sample" else None
    per_row = 4
    for start in range(0, len(samples), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, smp in zip(cols, samples[start:start + per_row]):
            with col:
                st.image(smp["path"], width="stretch")
                label = DISPLAY_NAMES.get(smp["true"], "Unlabelled") if smp["true"] else "Unlabelled"
                st.markdown(f'<div class="sample-caption">{label}</div>', unsafe_allow_html=True)
                is_current = smp["path"] == current
                st.button(
                    "Showing" if is_current else "Classify this scan",
                    key=f"btn_{smp['path']}",
                    on_click=use_sample,
                    args=(smp["path"],),
                    disabled=is_current,
                )

st.markdown(
    '<p class="footer-note">For research and education only. This is not a medical device and must not '
    "be used for diagnosis.</p>",
    unsafe_allow_html=True,
)
