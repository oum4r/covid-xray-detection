# COVID-19 X-Ray Detection System 🩻🤖

[![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-orange?logo=mathworks)](https://www.mathworks.com/products/matlab.html)
[![Deep Learning Toolbox](https://img.shields.io/badge/Toolbox-Deep%20Learning-blue)](https://www.mathworks.com/products/deep-learning.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success)](#-results)

Automated **COVID-19 diagnosis from chest X-ray images** using Convolutional Neural Networks (CNNs) and **transfer learning (VGG19)**.  
This project achieved **95% test accuracy**, demonstrating the potential of AI-driven medical imaging.

---

## 🌟 Key Highlights

- ✅ Achieved **95% classification accuracy**
- 🧠 Compared **baseline CNNs** with **VGG19 transfer learning**
- 🔄 Built **preprocessing + augmentation pipeline**
- 📊 Evaluated with **confusion matrices and validation tracking**

---

## 🔍 Overview

- **Objective**: Classify chest X-rays into 4 categories (COVID-19, Normal, Pneumonia, Lung Opacity).  
- **Approach**:
  - Data augmentation (rotation, scaling, reflection, shear, translation).
  - CNN baseline model (custom).
  - Transfer learning with **VGG19** for improved performance.
- **Tools**: MATLAB (Deep Learning Toolbox, Image Processing Toolbox).

---

## 📂 Repository Structure
```
COVID19-XRay-Detection-System/
│── src/ # MATLAB source code
│── data/ # Dataset placeholder
│── results/ # Experimental results
│── docs/ # Technical documentation
│── requirements.txt # Dependencies list
│── .gitignore # Git ignore rules
│── README.md # Project overview (this file)
```

---

## ⚙️ Installation

### Requirements
- MATLAB **R2024b** (or later)
- Deep Learning Toolbox
- Image Processing Toolbox
- Signal Processing Toolbox

⚠️ Note: Experiment 3 requires the **Deep Learning Toolbox Model for VGG-19 Network** add-on.  
Install it from MATLAB Add-On Explorer before running.

## 🚀 Usage
**1) Clone the repository**
```bash
git clone https://github.com/yourusername/COVID19-XRay-Detection-System.git
cd COVID19-XRay-Detection-System
```

**2) Place the dataset**

Create the data/ folder and arrange images by class (do not commit the raw data to Git):
```text
data/
├─ COVID/
├─ Normal/
├─ Lung_Opacity/
└─ Viral_Pneumonia/
```

Full provenance and citations are in data/README.md

**3) Run the project in MATLAB**

Open MATLAB at the repo root, then:
```matlab
addpath(genpath('src'));
cd src
main
```

**(Optional) Run a specific experiment**
```matlab
% Baseline CNN
[imdsTrain, imdsVal, imdsTest, YTest] = loadData('../data');
experiment1(imdsTrain, imdsVal, imdsTest, YTest);

% Balanced CNN
[imdsTrain, imdsVal, imdsTest, YTest] = loadData('../data');
experiment2(imdsTrain, imdsVal, imdsTest, YTest);

% VGG19 Transfer Learning
[imdsTrain, imdsVal, imdsTest, YTest] = loadData('../data');
experiment3(imdsTrain, imdsVal, imdsTest, YTest);
```

## 📂 Dataset

This project uses the COVID-19 Chest X-Ray Database (Qatar University, University of Dhaka, and collaborators).
Composition (latest release cited by the authors):
- **COVID-19**: 3,616
- **Normal**: 10,192
- **Lung Opacity (non-COVID)**: 6,012
- **Viral Pneumonia**: 1,345

All images are in PNG format with a resolution of 299×299 pixels. Sources include RSNA, PadChest, Kaggle, Eurorad, SIRM, Figshare, and others.
See full details and required citations in data/README.md

## 📊 Results
- **Baseline CNN**: ~85% accuracy
- **Balanced CNN (oversampling + augmentation)**: ~90% accuracy
- **Transfer Learning (VGG19)**: ~95% accuracy

Confusion matrices (saved/shown during runs) demonstrate robust class-wise performance.

## 📌 Future Work

- Evaluate additional backbones (ResNet, DenseNet, EfficientNet).
- Explore class-imbalance strategies (focal loss, class weights).
- Expand dataset and add cross-institutional validation.
- Package as a MATLAB App or Python inference service for demo purposes.
