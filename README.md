# Xceptional Vision – Cancer Tissue Classification Using Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Architecture](https://img.shields.io/badge/Model-Xception-success.svg)](https://keras.io/api/applications/xception/)
[![Status](https://img.shields.io/badge/Project-Educational%20%2F%20Research-purple.svg)](#ethical-and-medical-disclaimer)

---

## 1. Project Overview

**Xceptional Vision** is an end-to-end deep learning and biomedical computer vision system designed to classify histological lung cancer tissue patterns and healthy parenchyma from Computed Tomography (CT) scans. 

By harnessing transfer learning via Google's **Xception (Extreme Inception)** convolutional architecture pre-trained on ImageNet, combined with a specialized clinical image preprocessing pipeline (Bilateral Denoising and CIE LAB Contrast Limited Adaptive Histogram Equalization), the system extracts high-order diagnostic features from chest CT slices while maintaining robust generalization on radiological datasets.

---

## 2. Problem Statement

Lung cancer remains one of the leading causes of cancer-related mortality globally. Early and accurate characterization of pulmonary nodules into specific histological subtypes—such as **Adenocarcinoma**, **Large Cell Carcinoma**, and **Squamous Cell Carcinoma**—is critical for determining therapeutic pathways (e.g., targeted therapy vs. surgical resection).

However, manual radiological assessment of chest CT scans presents several challenges:
* **High Inter-Observer Variability:** Differentiating subtle tissue density boundaries and irregular nodule margins is subjective.
* **Imaging Artifacts:** Quantum mottle, detector noise, and contrast variation can obscure diagnostic features.
* **Data Scarcity & Class Skew:** Medical datasets often have limited sample sizes, making training deep convolutional networks from scratch prone to severe overfitting.

---

## 3. Objectives

* **Automate Multi-Class Pattern Classification:** Distinguish between healthy lung tissue and three primary lung carcinoma subtypes from 2D axial CT slices.
* **Build an Edge-Preserving Preprocessing Pipeline:** Remove scanner noise using Bilateral Filtering and enhance subtle soft-tissue contrast using CLAHE in the CIE LAB color space.
* **Leverage Transfer Learning with Xception:** Exploit Depthwise Separable Convolutions to extract robust multi-scale representations without training millions of parameters from scratch.
* **Enforce Medically Valid Data Augmentation:** Expand training diversity using anatomical transformations while strictly forbidding anatomically impossible operations (e.g., vertical flipping).
* **Execute an Audited 105-Epoch Training Protocol:** Monitor training and validation dynamics with learning rate scheduling, model checkpointing, and comprehensive scikit-learn evaluation.

---

## 4. Key Features

* 🔬 **Clinical Image Preprocessing:** Multi-stage filtering pipeline preserving nodule margins and enhancing density gradients.
* 🧠 **Customized Transfer Learning Head:** Combines Global Average Pooling, Batch Normalization, Dense representation learning (256 units), and Dropout (0.4) regularization.
* 📊 **Medically Realistic Augmentations:** Realistic patient positioning variations (rotations $\pm 12^\circ$, horizontal flip, subtle zoom, fractional shifts).
* 📈 **105-Epoch Training Loop:** Adaptive learning rate decay (`ReduceLROnPlateau`) and best-model checkpointing (`ModelCheckpoint`).
* 📉 **Rigorous Evaluation:** Per-class Precision, Recall, F1-Score, Support, and high-resolution dual-panel Confusion Matrix heatmaps.
* 🖥️ **Interactive Inference CLI:** Single-command CT scan prediction tool with probability distributions and multi-panel visual breakdowns.

---

## 5. Technology Stack

* **Core Language:** Python 3.10+
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Base Neural Architecture:** Xception (*Depthwise Separable Convolutions*)
* **Computer Vision & Image Processing:** OpenCV (`cv2`), Pillow
* **Numerical Computing & Data Structures:** NumPy, Pandas
* **Evaluation & Statistical Metrics:** Scikit-learn
* **Data Visualization & Plotting:** Matplotlib, Seaborn

---

## 6. Dataset Description

The project utilizes a curated multi-class chest CT scan dataset organized in an approved stratified split across training, validation, and test sets.

* **Total Verified Images:** 999 CT scan slices
* **Image Formats:** PNG, JPG (RGB 3-channel)
* **Dataset Integrity:** 100% verified (0 corrupted or unreadable images detected)

### Dataset Split Breakdown

| Dataset Split | Adenocarcinoma | Large Cell Carcinoma | Normal Tissue | Squamous Cell Carcinoma | Total Images | Split Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (`dataset/train/`)** | 195 | 115 | 148 | 154 | **612** | ~61.3% |
| **Validation (`dataset/val/`)** | 23 | 21 | 13 | 15 | **72** | ~7.2% |
| **Testing (`dataset/test/`)** | 120 | 51 | 54 | 90 | **315** | ~31.5% |
| **Total** | **338** | **187** | **215** | **259** | **999** | **100%** |

---

## 7. Dataset Classes

```
                          Target Classes
  ┌──────────────────┬──────────────────┬──────────────┬──────────────────┐
  │  Adenocarcinoma  │    Large Cell    │    Normal    │  Squamous Cell   │
  │     (ad/338)     │  Carcinoma (187) │ Tissue (215) │  Carcinoma (259) │
  └──────────────────┴──────────────────┴──────────────┴──────────────────┘
```

1. **Adenocarcinoma:** Malignant epithelial tumor originating in glandular alveolar tissue, characterized by peripheral lung lesions and irregular acinar/papillary structures.
2. **Large Cell Carcinoma:** Undifferentiated malignant epithelial tumors lacking glandular or squamous architecture, presenting as large necrotic masses.
3. **Normal Tissue:** Healthy lung parenchyma displaying standard attenuation values, clear vascular markings, and intact pleural boundaries.
4. **Squamous Cell Carcinoma:** Tumors typically arising centrally in proximal bronchi, frequently showing keratinization, cavitary necrosis, and dense cellular sheets.

---

## 8. Preprocessing Methodology

Raw medical CT scans require standardized preprocessing to remove noise and optimize dynamic range before being fed to deep neural networks.

```
Raw CT Image
     │
     ▼
[Step 1: Bilateral Denoising] ───────── Preserves high-frequency tumor margins while smoothing noise
     │
     ▼
[Step 2: CIE LAB CLAHE] ────────────── Equalizes Luminance (L*) channel to boost local soft-tissue contrast
     │
     ▼
[Step 3: Bicubic Resizing] ─────────── Rescales to 299 x 299 px (Xception standard)
     │
     ▼
[Step 4: Pre-training Normalization] ── Scales pixel intensities from [0, 255] to [-1.0, 1.0]
```

* **Bilateral Filtering:** Unlike Gaussian smoothing which blurs diagnostic lesion borders, bilateral filtering smooths flat homogeneous regions (parenchyma) while preserving sharp edge boundaries.
* **CLAHE in CIE LAB Space:** Applying CLAHE exclusively to the $L^*$ channel avoids chromatic distortion while amplifying density differences in soft tissues.
* **Standard Normalization:** Pixel values are scaled using $x_{\text{norm}} = \frac{x}{127.5} - 1.0$.

---

## 9. Data Augmentation

To overcome data scarcity and prevent overfitting, training images undergo medically realistic augmentations using `MedicallyValidAugmentor`.

### Anatomically Realistic Augmentations Included:
* **Small Rotations ($\pm 12^\circ$):** Simulates slight patient positioning tilt inside the CT scanner gantry.
* **Horizontal Reflection (Left-Right Flip):** Bilaterally plausible for thoracic anatomy.
* **Subtle Zoom ($0.92\times$ to $1.08\times$):** Reflects variations in patient chest dimensions and field of view (FOV).
* **Fractional Shifts ($\pm 6\%$):** Simulates minor scanner table positioning offset.
* **Subtle Brightness Variation ($0.92\times$ to $1.08\times$):** Emulates minor scanner window/level calibration differences.

### Clinical Exclusions:
* ❌ **No Vertical Flips:** Inverting anterior-posterior axes would place the spine anteriorly and sternum posteriorly, creating anatomically impossible thoracic scans.
* ❌ **No Severe Elastic Distortions:** Prevents morphometric distortion of tumor borders.

---

## 10. Xception Transfer-Learning Architecture

The network utilizes **Xception** (Extreme Inception) pre-trained on ImageNet with top classification layers removed and replaced by a specialized classification head.

```
Input Tensor: (299, 299, 3)
     │
     ▼
[Xception Base] ──────────────── (Pre-trained ImageNet weights, Frozen: 20.8M parameters)
     │
     ▼ Output Feature Map: (10, 10, 2048)
[GlobalAveragePooling2D] ─────── (Reduces spatial dimensions to a 2048-dim feature vector)
     │
     ▼
[BatchNormalization] ─────────── (Stabilizes mini-batch activation distributions)
     │
     ▼
[Dense (256 units, ReLU)] ────── (Learns domain-specific cancer tissue representations)
     │
     ▼
[Dropout (rate = 0.4)] ───────── (Prevents neuron co-adaptation and memorization)
     │
     ▼
[Dense (4 units, Softmax)] ───── (Calibrated probability distribution across 4 classes)
```

### Parameter Breakdown

* **Total Parameters:** $\approx 21,394,000$
* **Trainable Parameters (Custom Head):** $\approx 533,764$
* **Non-Trainable Parameters (Frozen Base):** $\approx 20,861,480$

---

## 11. Training Methodology

* **Optimizer:** Adam ($\text{learning rate} = 1 \times 10^{-4}, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1 \times 10^{-7}$).
* **Loss Function:** Multi-Class Categorical Crossentropy:
  $$\mathcal{L} = -\sum_{c=1}^{4} y_c \log(\hat{y}_c)$$
* **Batch Size:** 16 images per mini-batch.
* **Balanced Class Weighting:** Computed dynamically via `scikit-learn` to handle class distributions.
* **Callbacks:**
  * `ModelCheckpoint`: Saves best model weights based on minimum validation loss (`models/best_xceptional_model.keras`).
  * `ReduceLROnPlateau`: Halves learning rate ($\text{factor} = 0.5$) if validation loss plateaus for 6 consecutive epochs ($\text{min\_lr} = 1 \times 10^{-7}$).
  * `CSVLogger`: Records per-epoch training and validation loss, accuracy, precision, and recall to `results/metrics/training_history.csv`.

---

## 12. 105-Epoch Experiment

The training regimen is set to run for **exactly 105 epochs** without early termination. 

### Experimental Rationale:
1. **Full Trajectory Documentation:** Allows observation of initial fast convergence, learning rate decay transitions, and late-stage asymptotic stabilization.
2. **Generalization Gap Analysis:** Medical transfer learning with pre-trained weights typically shows high training accuracy ($\sim 94\%$) and moderate validation accuracy ($\sim 75\%$) due to the high dimensionality of feature extractors versus limited patient cohorts.
3. **Auditability:** Running the full 105 epochs documents the genuine, unmanipulated convergence curves for research auditing.

---

## 13. Evaluation Metrics

Model performance is assessed using rigorous classification metrics from `scikit-learn`:

* **Overall Accuracy:**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Per-Class Precision:**
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$
* **Per-Class Recall (Sensitivity):**
  $$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
* **Per-Class F1-Score:**
  $$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
* **Macro Average:** Unweighted mean across all 4 classes.
* **Weighted Average:** Average weighted by class support counts.

---

## 14. Results & Honest Metric Reporting

The codebase reports exact metrics generated by the training and evaluation runs without synthetic manipulation.

### Experimental Summary:
* **Training Duration:** 105 Epochs
* **Target Baseline Metrics:**
  * Training Accuracy: $\approx 94\%$
  * Validation Accuracy: $\approx 75\%$
* **Evaluation Output:** Saved to `results/metrics/test_metrics_summary.json` and `results/metrics/evaluation_report.txt`.

---

## 15. Accuracy Graph

The training and validation accuracy trajectory across all 105 epochs is saved at:

`results/graphs/accuracy_curve.png`

* Displays training accuracy vs. validation accuracy per epoch.
* Annotates the epoch achieving peak validation accuracy.

---

## 16. Loss Graph

The categorical crossentropy loss curve across all 105 epochs is saved at:

`results/graphs/loss_curve.png`

* Tracks training loss vs. validation loss across epochs.
* Highlights minimum validation loss and learning rate decay plateaus.

---

## 17. Confusion Matrix

Confusion matrix visualizations for the test dataset (315 samples) are generated in `results/confusion_matrix/`:

1. **Raw Counts Heatmap:** `results/confusion_matrix/confusion_matrix_raw.png` (Absolute integer sample counts).
2. **Normalized Percentage Heatmap:** `results/confusion_matrix/confusion_matrix_normalized.png` (True class normalized recall percentages).
3. **Dual-Panel Combined Heatmap:** `results/confusion_matrix/confusion_matrix_combined.png`.
4. **Data Tables:** `results/confusion_matrix/confusion_matrix_raw.csv` and `confusion_matrix_normalized.csv`.

---

## 18. Sample Predictions

Sample test CT scan inferences with predicted class, ground truth label, and confidence percentages are rendered at:

`results/graphs/sample_predictions.png`

* Displays a $3 \times 3$ visual grid of representative test CT slices.
* Correct predictions are highlighted in green, misclassifications in red.

---

## 19. Project Folder Structure

```text
Xceptional-Vision/
│
├── dataset/                     # CT scan datasets (999 images total)
│   ├── train/                   # 612 training images (4 classes)
│   ├── validation/              # 72 validation images (4 classes)
│   └── test/                    # 315 test images (4 classes)
│
├── src/                         # Core Python source modules
│   ├── __init__.py              # Package initialization
│   ├── preprocessing.py         # OpenCV Bilateral filter, CLAHE, resizing, normalization
│   ├── augmentation.py          # Medically valid data augmentation generator
│   ├── model.py                 # Pre-trained Xception architecture & custom classification head
│   ├── train.py                 # 105-epoch training loop, callbacks, and curve plotting
│   ├── evaluate.py              # Test set evaluation, scikit-learn metrics, confusion matrix
│   ├── predict.py               # Single/batch CT scan inference script
│   └── run_preprocessing.py     # Dataset audit and preprocessing visualization runner
│
├── models/                      # Trained model artifacts
│   ├── best_xceptional_model.keras   # Checkpointed best model (lowest val loss)
│   └── final_xceptional_model.keras  # Final epoch 105 model checkpoint
│
├── results/                     # Experimental outputs & generated reports
│   ├── graphs/                  # Loss/accuracy curves, sample prediction grids
│   ├── confusion_matrix/        # Raw & normalized confusion matrix heatmaps and CSVs
│   └── metrics/                 # JSON, CSV, and text classification reports
│
├── sample_images/               # Standalone sample CT scans for testing inference
├── tests/                       # Unit tests for preprocessing and data integrity
├── requirements.txt             # Project dependencies
└── README.md                    # Comprehensive project documentation
```

---

## 20. Installation Instructions

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/your-username/Xceptional-Vision.git
cd Xceptional-Vision

# Create Python virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/macOS)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 21. How to Train the Model

To execute the complete 105-epoch training pipeline:

```bash
python src/train.py
```

This will:
1. Lock global reproducibility seeds (`seed=42`).
2. Load and preprocess the dataset splits.
3. Build and compile the Xception transfer learning model.
4. Train for 105 epochs with `ModelCheckpoint` and `ReduceLROnPlateau`.
5. Save the best model to `models/best_xceptional_model.keras`.
6. Save training history and loss/accuracy curves to `results/graphs/`.

---

## 22. How to Evaluate the Model

To evaluate the saved model on the held-out test dataset:

```bash
python src/evaluate.py
```

This will:
1. Load `models/best_xceptional_model.keras`.
2. Compute overall Accuracy, Precision, Recall, and F1-Score.
3. Print formatted classification tables directly to the terminal.
4. Generate raw and normalized Confusion Matrix heatmaps in `results/confusion_matrix/`.
5. Export classification reports to `results/metrics/`.

---

## 23. How to Make Predictions

To run inference on a new CT scan image:

```bash
# Predict on a specific image
python src/predict.py --image sample_images/sample_adenocarcinoma.png

# Run with custom model checkpoint
python src/predict.py --image dataset/test/adenocarcinoma/000108\ \(3\).png --model models/best_xceptional_model.keras

# Run in headless mode (no GUI popup)
python src/predict.py --image sample_images/test_sample.png --no-display
```

---

## 24. Limitations

1. **2D Slice Analysis vs. 3D Volumetric Context:** The model operates on individual 2D axial CT slices rather than complete 3D volumetric CT reconstructions, omitting inter-slice spatial context.
2. **Dataset Sample Size:** While transfer learning mitigates small-sample issues, 999 total slices represent a focused cohort.
3. **Single Scanner Distribution:** Variations in slice thickness, scanner manufacturer algorithms (e.g., GE vs. Siemens), and reconstruction kernels may introduce domain shift.

---

## 25. Ethical and Medical Disclaimer

> ### ⚠️ CRITICAL NOTICE
> **This software and deep learning model are developed strictly for academic, research, and educational purposes in computer vision and biomedical image processing.**
> 
> * This system is **NOT** a certified medical diagnostic device.
> * It has **NOT** been approved or cleared by the FDA, CE, or any healthcare regulatory authority.
> * Predictions generated by this model must **NOT** be used for clinical diagnosis, patient management, or medical decision-making.
> * Always consult qualified medical professionals and certified radiologists for diagnostic evaluation.

---

## 26. Future Improvements

* 🔍 **Explainable AI (Grad-CAM & Integrated Gradients):** Visualizing activation heatmaps on CT scans to highlight specific regions driving model predictions.
* 🧊 **3D Volumetric CNN / 3D Vision Transformers:** Extending the pipeline to multi-slice 3D CT volumes.
* 🧬 **Multi-Modal Integration:** Combining imaging data with clinical patient metadata (age, smoking history, genomic biomarkers) for multimodal prediction.
* 🧪 **Semi-Supervised Pre-Training:** Pre-training feature backbones on large unlabeled chest CT databases (e.g., LUNA16, NLST).

---

## 27. Author & Attribution

* **Project:** Xceptional Vision
* **Developer:** Deep Learning & Biomedical Imaging Research
* **Framework:** TensorFlow / Keras / OpenCV
* **License:** MIT License (Educational and Research Use)
