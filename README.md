# Image Classification with Convolutional Neural Networks (CNN)

## Project Overview

This project implements image classification on the CIFAR-10 dataset using various deep learning approaches, culminating in a transfer learning model based on ResNet50. The project demonstrates the progression from basic Multi-Layer Perceptrons (MLPs) to Convolutional Neural Networks (CNNs) and finally to transfer learning, showcasing the improvements in classification accuracy at each stage.

## Dataset

- *CIFAR-10*: 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- 50,000 training images and 10,000 testing images.

## Implemented Models

### 1. Multi-Layer Perceptron (MLP)

- Multiple architectures tested with varying layers and neurons.
- Best performance: ~51.26% accuracy on the test set.

### 2. Convolutional Neural Network (CNN)

- Custom CNN architecture implemented using TensorFlow and Keras.
- Includes convolutional layers, max pooling, dropout, and dense layers.
- Performance: ~68.21% accuracy on the test set.

### 3. Transfer Learning with ResNet50

- Pre-trained ResNet50 model used as a base.
- Custom classification layers added on top.
- Input images upsampled from 32x32 to 224x224.
- Data augmentation applied using ImageDataGenerator.
- Fine-tuning performed on the last 30 layers of ResNet50.
- Final performance: >91.4% accuracy on the test set.

## Key Features

- Data normalization and preprocessing.
- Implementation of data augmentation techniques.
- Utilization of transfer learning for improved performance.
- Learning rate scheduling and early stopping for optimal training.
- Visualization of model predictions.

## Technologies Used

- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Model Architecture (Transfer Learning)

- Base: ResNet50 (pre-trained on ImageNet)
- Additional layers:
    - Global Average Pooling
    - Dense layers with ReLU activation
    - Batch Normalization and Dropout for regularization
    - Final Dense layer with softmax activation for 10-class classification

## Training Process

- Optimizer: Adam
- Loss function: Categorical Cross-entropy
- Callbacks: ReduceLROnPlateau, EarlyStopping
- Batch size: 64
- Epochs: 15 (with early stopping)

## Results

- MLP: ~51.26% accuracy
- CNN: ~68.21% accuracy
- Transfer Learning (ResNet50): >91.4% accuracy

## Conclusion

The project demonstrates the effectiveness of transfer learning in achieving high accuracy on the CIFAR-10 dataset. The progression from MLP to CNN to transfer learning showcases the improvements in model performance as more sophisticated techniques are applied.
