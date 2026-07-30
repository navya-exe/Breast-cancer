# Breast Cancer Detection using CNN

A convolutional neural network that flags Invasive Ductal Carcinoma (IDC) regions in breast histopathology whole-slide images, benchmarked against a traditional ML baseline on the same data.

## Overview

Traditional mammography-based diagnosis relies on expert radiologists and is prone to false positives/negatives. This project trains a CNN classifier on histopathology image patches to automate IDC detection, and compares its performance directly against a traditional machine learning pipeline.

## Dataset

- **Source:** [CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) and [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (Kaggle)
- **Size:** ~275,000 labeled RGB image patches, 50×50 pixels each
- **Labels:** IDC-positive vs. IDC-negative tissue regions

## Method

- Data cleaning and exploratory analysis on the DICOM metadata and case-description CSVs
- Image patches loaded, balanced, and split into train/test sets
- Data augmentation via `ImageDataGenerator`
- A CNN built with stacked `Conv2D → BatchNormalization → MaxPooling` blocks, a dense classification head with dropout, trained with `EarlyStopping` and a learning-rate plateau callback
- Evaluated with accuracy, confusion matrix, and predicted-vs-true class comparison

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])
```

## Results

| Model | Accuracy |
|---|---|
| Traditional ML baseline | 78% |
| **CNN (this project)** | **87%** |

The CNN approach improved on the traditional ML baseline by 9 percentage points.

## Repository Structure

```
breast-cancer-detection/
├── notebooks/
│   └── breast_cancer_detection_using_cnn.ipynb   # full pipeline: data cleaning, EDA, model, evaluation
├── models/
│   └── CNN_model.h5                              # trained model weights
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
```

Download the datasets from Kaggle (links above), place them under a `content/` directory as referenced in the notebook, then run `notebooks/breast_cancer_detection_using_cnn.ipynb` top to bottom.

To use the trained model directly instead of retraining:

```python
import tensorflow as tf
model = tf.keras.models.load_model('models/CNN_model.h5')
```

## Author

Ch. Navya Naidu
