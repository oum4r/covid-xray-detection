# Chest-X-ray-Classification-with-VGG19-Transfer-Learning
This project applies deep learning to chest X-ray image classification, including the detection of COVID-19 cases. Using transfer learning with VGG19, the model achieved high accuracy on a curated dataset of medical images.

🌟 Highlights
- Achieved 95% test accuracy using VGG19 Transfer Learning.
- Built a preprocessing pipeline (resizing, greyscale conversion, augmentation).
- Compared baseline CNNs with transfer learning for performance gains.
- Evaluated models using confusion matrices and validation accuracy tracking.

🔍 Overview
- Objective: Classify chest X-ray images into 4 classes, including COVID and pneumonia categories.
- Approach:
  - Image preprocessing and augmentation (rotation, reflection, scaling).
  - Implementation of VGG19 transfer learning.
  - Benchmarking against simpler CNN topologies.
- Tools: MATLAB, Deep Learning Toolbox, Signal Processing Toolbox.

📂 Dataset
- Chest X-ray dataset provided as part of the coursework specification.
- Images standardised to 128x128x1 for training.
- Training set augmented to improve generalisation.

📊 Results
- VGG19 Transfer Learning significantly outperformed baseline CNNs.
- Final model reached 95% accuracy on test data.
- Confusion matrices confirmed robust classification across classes.

📌 Future Work
- Extend experiments with other pretrained models (ResNet, DenseNet).
- Explore hybrid CNN-LSTM approaches for temporal medical data.
- Expand dataset for wider applicability.
