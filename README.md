# Brain Tumor MRI Classification — CNN vs Vision Transformer

Multi-class classification of brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**, benchmarking three deep-learning architectures — **VGG16**, **ResNet50**, and **Vision Transformer (ViT-B/16)** — on the same dataset, split, and evaluation protocol, with the trained model packaged as a Dockerized Streamlit web app.

\---

## Highlights



\[!\[Open in Streamlit](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://brain-tumor-classifier.streamlit.app)



🔗 \*\*Live demo:\*\* https://brain-tumor-classifier.streamlit.app



## Dataset

**Brain Tumor MRI Dataset** — [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

|Split|Images|
|-|-|
|Training (80% of train folder)|4,480|
|Validation (20% of train folder)|1,120|
|Test (held-out folder)|1,600|

Four balanced classes with 1,400 training images each:

```
glioma  |  meningioma  |  notumor  |  pituitary
```

Class index mapping used across all models: `{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}`

Downloaded programmatically so the notebooks are reproducible without manual setup:

```python
import kagglehub
path = kagglehub.dataset\_download("masoudnickparvar/brain-tumor-mri-dataset")
```

\---

## Repository Structure

```
.
├── VGG16\_Brain\_tumer\_Project.ipynb      # VGG16 fine-tuning + ViT built from scratch (Keras)
├── Resnet50\_\_ViT\_brain\_project.ipynb    # ResNet50 two-phase transfer learning + ViT-B/16 (PyTorch/HF)
├── app.py                               # Streamlit inference app
├── Dockerfile                           # Container build for the web app
├── requirements.txt                     # Runtime dependencies
└── README.md
```

\---

## Approach

All three models share an identical data pipeline — 224×224 input, an 80/20 stratified train/validation split from the training folder, a completely held-out test folder, and the same augmentation family (rotation, zoom, shift, horizontal flip). Only the architecture and its normalization scheme change, so the comparison is apples-to-apples.

### 1\. VGG16 — Partial Fine-Tuning

A classic CNN baseline. ImageNet weights are loaded without the classifier head, the earlier convolutional blocks stay frozen, and the **last 20 layers are unfrozen** so the deeper, task-specific filters can adapt to MRI texture while the generic edge/shape detectors stay intact.

```python
base\_model = VGG16(weights='imagenet', include\_top=False, input\_shape=(224,224,3))
base\_model.trainable = True
for layer in base\_model.layers\[:-20]:
    layer.trainable = False

model = Sequential(\[
    base\_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```

* Preprocessing: `rescale=1./255`
* Optimizer: Adam @ `1e-4`
* Loss: categorical cross-entropy
* 10 epochs

### 2\. ResNet50 — Two-Phase Transfer Learning

The strongest CNN in the comparison, trained in two deliberate stages instead of one.

**Phase 1 — train the head.** The entire ResNet50 backbone is frozen and only the new classification head learns, at a high learning rate (`1e-3`). This prevents large, randomly-initialized gradients from destroying the pretrained features on the first few batches.

**Phase 2 — fine-tune the top block.** Layers from index 143 onward are unfrozen and the whole thing is retrained at `1e-5` — a 100× lower learning rate — so the pretrained representations are nudged toward MRI rather than overwritten.

```python
# Phase 2
base\_model.trainable = True
for layer in base\_model.layers\[:143]:
    layer.trainable = False

model.compile(optimizer=Adam(learning\_rate=1e-5),
              loss='categorical\_crossentropy', metrics=\['accuracy'])
```

Head architecture: `GlobalAveragePooling2D → BatchNormalization → Dense(256, relu) → Dropout(0.5) → Dense(4, softmax)`

**One detail that materially affects results:** ResNet50 uses `preprocessing\_function=preprocess\_input` rather than `rescale=1./255`. The ImageNet weights were trained with a specific channel-wise mean subtraction, and feeding them naively-scaled `\[0,1]` images silently degrades the transferred features.

Callbacks: `EarlyStopping(patience=5, restore\_best\_weights=True)` and `ReduceLROnPlateau(factor=0.3, patience=3, min\_lr=1e-7)`, both monitoring validation loss.

### 3\. Vision Transformer (ViT-B/16) — PyTorch + Hugging Face

A pretrained transformer to test whether global self-attention beats convolutional inductive bias on this task. Checkpoint: `google/vit-base-patch16-224-in21k`.

```python
model = ViTForImageClassification.from\_pretrained(
    "google/vit-base-patch16-224-in21k",
    num\_labels=4,
    id2label={i: c for i, c in enumerate(class\_labels)},
    label2id={c: i for i, c in enumerate(class\_labels)},
    ignore\_mismatched\_sizes=True
).to(device)
```

Same two-phase strategy as ResNet50, ported to PyTorch:

|Phase|Trainable|Epochs|LR|
|-|-|-|-|
|1|Classifier head only (`model.vit` frozen)|10|`1e-3`|
|2|Full backbone unfrozen|15|`2e-5`|

* Optimizer: `AdamW`
* Loss: cross-entropy with `label\_smoothing=0.1`
* Normalization pulled from the checkpoint's own `ViTImageProcessor`, so the stats always match the pretrained weights
* A custom training loop reimplements early stopping, best-weight restoration, and `ReduceLROnPlateau` (Keras's `.fit()` conveniences aren't available in raw PyTorch)

A subtle correctness fix worth noting: after `random\_split`, the validation subset would otherwise inherit the *training* transforms — including augmentation. A clean `ImageFolder` with eval-only transforms is swapped in so validation measures the model, not the augmentation noise.

```python
val\_ds.dataset = datasets.ImageFolder(train\_path, transform=eval\_transform)
```

### 4\. Vision Transformer — Built From Scratch (Keras)

For comparison against the pretrained ViT, a transformer is also implemented from first principles in Keras — no pretrained weights — to demonstrate the architecture end to end:

* A custom `Patches` layer using `tf.image.extract\_patches` to split each 224×224 image into 196 non-overlapping 16×16 patches
* A custom `PatchEncoder` layer: linear projection to a 64-dim embedding **plus a learned positional embedding**, since attention is permutation-invariant and would otherwise be blind to spatial layout
* 4 transformer blocks, each with `LayerNormalization → MultiHeadAttention(4 heads) → residual → LayerNormalization → GELU MLP → residual`
* MLP classification head with dropout, trained with `AdamW` (weight decay `1e-4`) and tracked on accuracy, precision, and recall

This one trains for up to 50 epochs with early stopping (`patience=8`) and model checkpointing — from-scratch transformers are data-hungry and converge far more slowly than a fine-tuned CNN, which is exactly the point of including it.

\---

## Evaluation

Every model is scored on the same **1,600-image held-out test set**, never seen during training or validation, using:

* Test loss and accuracy
* Per-class **precision, recall, and F1** via `classification\_report`
* A **confusion matrix** heatmap, which is where the interesting failure modes show up — glioma and meningioma are the pair most often confused, since they share overlapping intensity and location patterns on MRI
* Training vs. validation accuracy/loss curves across both phases, stitched together to make the fine-tuning jump visible

\---

## Deployment

The best-performing model is served through a **Streamlit** web app and containerized for reproducible deployment.

**Dockerfile** (Python 3.12-slim base, TensorFlow runtime, port 8501):

```bash
docker build -t brain-tumor-app .
docker run -p 8501:8501 brain-tumor-app
```

Then open `http://localhost:8501`, upload an MRI scan, and the app returns the predicted class with confidence scores.

> \*\*Note:\*\* the ViT-B/16 model is saved in Hugging Face PyTorch format (`save\_pretrained`), not `.keras`. Serving it through the same TensorFlow-based container requires either a PyTorch serving path or an ONNX conversion step.

\---

## Running Locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Then open either notebook in Google Colab (both were developed there and assume a GPU runtime):

* `VGG16\_Brain\_tumer\_Project.ipynb` — VGG16 + from-scratch ViT
* `Resnet50\_\_ViT\_brain\_project.ipynb` — ResNet50 + pretrained ViT-B/16

The ResNet50 notebook pulls the dataset automatically via `kagglehub`; no manual download required.

\---

## Tech Stack

|Category|Tools|
|-|-|
|Deep Learning|TensorFlow / Keras, PyTorch, Hugging Face Transformers|
|Architectures|VGG16, ResNet50, ViT-B/16, custom ViT|
|Data \& Metrics|NumPy, scikit-learn, Pillow, torchvision|
|Visualization|Matplotlib, Seaborn|
|Deployment|Streamlit, Docker|
|Environment|Google Colab (GPU), kagglehub|

\---

## Saved Models

|Model|Artifact|Format|
|-|-|-|
|VGG16|`VGG16\_Brain\_Tumor.keras`|Keras|
|ResNet50|`brain\_tumor\_resnet50.keras`|Keras|
|ViT-B/16|`brain\_tumor\_vit\_b16/`|Hugging Face (PyTorch)|
|ViT (scratch)|`best\_vit\_model.keras`|Keras|

\---

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis.

