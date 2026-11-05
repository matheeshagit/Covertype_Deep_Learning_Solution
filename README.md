# Forest Covertype Multi-Class Classification using Deep Learning (Keras/TensorFlow)

An end-to-end deep learning project showcasing model design, optimization, and evaluation on the benchmark UCI Forest Covertype Dataset. This repository demonstrates how to handle mixed-type tabular features, manage class imbalance, and train a Multi-Layer Perceptron (MLP) architecture using Keras.

## 📌 Project Overview

The goal of this project is to predict 7 distinct forest cover types based on cartographic and environmental features (elevation, slope, soil types, etc.) collected by the US Forest Service.

Rather than relying on tree-based models (e.g., XGBoost), this project explores the performance of a modern Multi-Layer Perceptron (MLP) by leveraging custom regularizations, non-linear activation functions, and dynamic learning rate scheduling.

### Key Highlights

Dataset Scale: Over 581,000 instances with 54 features (10 continuous, 44 binary/categorical).

Core Frameworks: TensorFlow / Keras, Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib.

Model Architecture: Custom MLP with ReLU activations, BatchNormalization, and Dropout regularization.

Optimization Strategy: Mini-batch training with Adam optimizer, EarlyStopping, and ReduceLROnPlateau callbacks.

Test Performance: Achieved ~93% overall accuracy on an untouched 15% test set.

### Preprocessing Strategy

Continuous Features (Columns 0–9): Scaled using StandardScaler to maintain gradient stability during backpropagation.

Binary Features (Columns 10–53): Retained as 0/1 indicator flags (Wilderness Areas & Soil Types).

Target Variable Shifting: Shifted 1-indexed target labels (1-7) to 0-indexed values (0-6) to align with standard sparse_categorical_crossentropy loss.

Stratified Splitting: Data split into 70% Train, 15% Validation, and 15% Test sets, preserving original class ratios across all splits.

### Loss & Metrics

Loss Function: sparse_categorical_crossentropy

Optimizer: adam (Default learning rate 0.001)

Evaluation Metric: Accuracy
