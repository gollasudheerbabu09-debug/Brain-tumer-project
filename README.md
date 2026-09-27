# Brain Tumor MRI Classification — CNN vs Vision Transformer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-APP-LINK)

🔗 **Live demo:** https://brain-tumer-project-6th8pzh9rntvoaeahflfhg.streamlit.app/

Multi-class classification of brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. Four architectures — **ViT-B/16**, **ResNet50**, **VGG16**, and a **Vision Transformer built from scratch** — are trained and evaluated on the same held-out test set, and the models are served in a Streamlit web app that is also packaged with Docker.





https://github.com/user-attachments/assets/0b23abe8-1b12-41bd-876a-79395b7af76b


---

## Highlights

- Fine-tuned a pretrained ** ResNet50** to **98.13% test accuracy** on 1,600 held-out MRI scans, outperforming Vision Transformer (ViT-B/16) (95.50%) and VGG16 (95.63%).
- Diagnosed a silent weight-loading failure in the first ViT run (every backbone weight was reported `MISSING`, so the model trained from random initialization and reached only 57%); switching to `timm` so the pretrained weights actually loaded raised accuracy to 95.13%.
- Exported the ViT from PyTorch to **ONNX with int8 quantization**, shrinking it from 343 MB to 87 MB with no measured accuracy loss, so it runs alongside the TensorFlow models within free-tier hosting limits.
- Deployed all models in a **Streamlit** app with model selection and per-class confidence scores, containerized with **Docker** and hosted on Streamlit Community Cloud.



---

## Results

All models are scored on the same **1,600-image held-out test set** (400 per class), never seen during training or validation.

| Model | Framework | Test Accuracy | Macro F1 | Glioma Recall |
|---|---|---|---|---|
| **ViT-B/16** (pretrained, fine-tuned) | PyTorch + timm | **95.13%** | **0.950** | 0.820 |
| ResNet50 (two-phase fine-tuning) | TensorFlow / Keras | 98.50% | 0.93 | 0.800 |
| VGG16 (two-phase fine-tuning) | TensorFlow / Keras | 95.63% | 0.924 | 0.775 |


**Per-class results — ViT-B/16 (best model):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.997 | 0.820 | 0.900 |
| Meningioma | 0.884 | 0.988 | 0.933 |
| No tumor | 0.952 | 1.000 | 0.976 |
| Pituitary | 0.988 | 0.998 | 0.993 |

**Key observations**

- **Pretraining matters most for transformers.** The same ViT idea goes from 68% trained from scratch to 95% when starting from pretrained weights. Transformers lack the built-in spatial assumptions of CNNs, so on a dataset of ~5,600 images they depend heavily on pretraining.
- **Glioma is the hardest class for every model.** Glioma precision is near-perfect (≥0.99) but recall is 0.78–0.82: when a model misses a glioma, it usually predicts meningioma. Validation accuracy (~99% for ViT-B/16) is much higher than test accuracy on this class, which points to a shift between the dataset's training and test glioma images rather than a model-specific flaw.
- **Correct preprocessing is not optional.** ResNet50 and VGG16 both use the exact ImageNet normalization their pretrained weights expect (`preprocess_input`) rather than a plain `/255` rescale.

---

## Dataset

**Brain Tumor MRI Dataset** — [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

| Split | Images |
|---|---|
| Training (80% of train folder) | 4,480 |
| Validation (20% of train folder) | 1,120 |
| Test (held-out folder) | 1,600 |

Four balanced classes: `glioma | meningioma | notumor | pituitary`

Class index mapping used across all models: `{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}`

The notebooks download the dataset automatically:

```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

---

## Approach

Every model uses 224×224 inputs, an 80/20 train/validation split of the training folder, and the same held-out test folder. Only the architecture, its preprocessing, and its training schedule change.

### 1. ViT-B/16 — Pretrained Vision Transformer (PyTorch + timm)

Checkpoint: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k pretrained, ImageNet-1k fine-tuned).

- Full fine-tuning with **discriminative learning rates**: `3e-5` for the pretrained backbone, `3e-4` for the new classification head
- `AdamW` (weight decay `0.05`), **1-epoch linear warmup + cosine decay**, gradient clipping at 1.0
- Cross-entropy with `label_smoothing=0.1`
- **Mixed-precision training** (`torch.autocast` + `GradScaler`) on a T4 GPU, ~50 s per epoch
- Validation uses a separate, augmentation-free view of the data; best weights selected on validation accuracy
- Reached 99.38% validation and **95.13% test** accuracy in 12 epochs

**Deployment detail:** the trained model is exported to ONNX and dynamically quantized to int8 (343 MB → 87 MB). ONNX fp32, ONNX int8, and PyTorch all gave identical accuracy on a 400-image test sample, so the app serves the int8 model with `onnxruntime` and does not need PyTorch installed.

### 2. ResNet50 — Two-Phase Transfer Learning (Keras)

**Phase 1 — train the head.** The ResNet50 backbone is frozen and only the new head learns at `1e-3`, so large gradients from the randomly initialized head don't damage the pretrained features.

**Phase 2 — fine-tune the top.** Layers from index 143 onward are unfrozen and trained at `1e-5`, 100× lower, so the pretrained features adapt to MRI instead of being overwritten.

- Head: `GlobalAveragePooling2D → BatchNormalization → Dense(256, relu) → Dropout(0.5) → Dense(4, softmax)`
- Preprocessing: `resnet50.preprocess_input`
- Callbacks: `EarlyStopping(patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.3, patience=3)`

### 3. VGG16 — Two-Phase Fine-Tuning (Keras)

- Head: `GlobalAveragePooling2D → Dropout(0.4) → Dense(256, relu) → Dropout(0.4) → Dense(4, softmax)`. Global average pooling instead of `Flatten` cuts the head from ~6.4M to ~130K parameters and reduces overfitting.
- Phase 1: backbone frozen, head trained at `1e-3` for 6 epochs
- Phase 2: blocks 4–5 unfrozen, fine-tuned at `1e-5` with early stopping on validation accuracy and learning-rate reduction
- Preprocessing: `vgg16.preprocess_input`; label smoothing `0.1`
- Fast `tf.data` pipeline with on-GPU augmentation (flip, rotation, zoom, translation, contrast), trained on a Kaggle GPU
- Improved on the first VGG16 version (91.94%, `/255` rescale with a `Flatten` head) to **92.63%**

### 4. Vision Transformer — Built From Scratch (Keras)

Implemented from first principles with no pretrained weights, to show the architecture end to end and to measure how much pretraining contributes:

- A custom `Patches` layer (`tf.image.extract_patches`) splits each image into 196 non-overlapping 16×16 patches
- A custom `PatchEncoder` layer: linear projection to 64 dimensions plus a **learned positional embedding**, since self-attention alone has no notion of patch position
- 4 transformer blocks: `LayerNorm → MultiHeadAttention(4 heads) → residual → LayerNorm → GELU MLP → residual`
- `AdamW` (weight decay `1e-4`), early stopping (`patience=8`), up to 50 epochs

It reaches 68.06% — far below the pretrained ViT-B/16, which is the expected result for a transformer trained from scratch on a small dataset.

---

## Web App

The Streamlit app lets you pick a model, upload an MRI scan, and see the predicted class with confidence scores for all four classes. Each model receives exactly the resizing and normalization it was trained with. Only one model is kept in memory at a time, so the app fits within free-tier hosting limits.

### Run with Docker

```bash
git lfs install
git clone https://github.com/gollasudheerbabu09-debug/Brain-tumer-project.git
cd Brain-tumer-project
docker build -t brain-tumor-app .
docker run -p 8501:8501 brain-tumor-app
```

Then open `http://localhost:8501`.

### Run without Docker

```bash
pip install -r requirements.txt
streamlit run app.py
```

> The model files are stored with **Git LFS**. Install Git LFS before cloning, or the model files will be small pointer files instead of real models.

---

## Repository Structure

```
.
├── app.py                               # Streamlit app
├── Dockerfile                           # Container build for the app
├── requirements.txt                     # App dependencies
├── vit_b16.onnx                         # ViT-B/16 (ONNX, int8)            — Git LFS
├── brain_tumor_resnet50.keras           # ResNet50                          — Git LFS
├── VGG16_Brain_Tumor_v2.keras           # VGG16                             — Git LFS
├── ViT_B16_Brain_Tumor.ipynb            # ViT-B/16 training + ONNX export (Colab)
├── Resnet50___ViT_brain_project.ipynb   # ResNet50 training (+ first ViT attempt)
├── VGG16_Brain_Tumor_v2.ipynb           # Improved VGG16 training (Kaggle)
├── VGG16_Brain_tumer_Project.ipynb      # First VGG16 + ViT from scratch
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, timm, TensorFlow / Keras |
| Architectures | ViT-B/16, ResNet50, VGG16, custom ViT |
| Model Export | ONNX, ONNX Runtime (int8 dynamic quantization) |
| Data & Metrics | NumPy, scikit-learn, Pillow, torchvision, tf.data |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit, Streamlit Community Cloud, Docker, Git LFS |
| Environment | Google Colab (T4 GPU), Kaggle (GPU), kagglehub |

---

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis.
