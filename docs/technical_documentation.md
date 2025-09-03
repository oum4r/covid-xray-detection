
---

## 📄 Technical Documentation Outline (docs/technical_documentation.md)

```markdown
# Technical Documentation: COVID-19 X-Ray Detection System

## 1. Introduction
- Motivation: AI-assisted COVID-19 screening.
- Dataset overview.

## 2. Data Preprocessing
- Splitting into train/val/test sets.
- Augmentation strategies (rotation, reflection, scaling, shear, translation).
- Greyscale conversion and resizing.

## 3. Model Architectures
### Experiment 1: Baseline CNN
- Custom CNN with increasing filter depth.
- Layers: Conv → BatchNorm → ReLU → Pooling.

### Experiment 2: Balanced CNN
- Oversampling smaller classes.
- Stronger augmentation strategies.

### Experiment 3: Transfer Learning (VGG19)
- Pretrained VGG19 base.
- Custom fully connected layers + dropout.

## 4. Training Setup
- Optimiser: Adam
- Learning rate schedules
- Mini-batch sizes
- Validation frequency

## 5. Evaluation
- Metrics: Accuracy, confusion matrices
- Final results summary

## 6. Limitations
- Dataset size constraints.
- Generalisation to clinical use.

## 7. Future Work
- ResNet, DenseNet, EfficientNet.
- Larger, diverse datasets.
