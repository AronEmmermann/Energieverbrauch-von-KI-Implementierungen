# Energieverbrauch-von-KI-Implementierungen
# Deep Learning Models: Training, Pruning, Quantization, and Transfer Learning

This repository contains various deep learning models and techniques, including training, pruning, quantization, and transfer learning, using TensorFlow and Keras. The repository focuses on the CIFAR-10 dataset and shows the accuracy in relation to the energy consumption of different models .

## Introduction

This repository provides implementations of various deep learning techniques applied to the CIFAR-10 dataset. The primary focus is on:

## Models

### Feedforward Neural Network (FFNN)

#### FFNN with Few Layers and Few Neurons
The implementation of a FFNN with a shallow architecture and fewer neurons per layer.

#### FFNN with Many Layers and Many Neurons
The implementation of a FFNN with a deep architecture and more neurons per layer.

### Convolutional Neural Network (CNN)

#### CNN with More Convolutional Layers
The implementation of a CNN with multiple convolutional layers for feature extraction.

#### CNN with Fewer Convolutional Layers
The implementation of a CNN with fewer convolutional layers but still effective for classification tasks.

### Transfer Learning

#### Pretrained Model with Fine-Tuning
Using a pretrained MobileNetV2 model and fine-tuning it on the CIFAR-10 dataset to improve performance.

### Pruning

#### Pruning 50% Weights
Applying pruning techniques to remove 50% of the weights in the model to reduce its size without significant loss of accuracy.

### Quantization

#### 8-bit Quantization
Implementing 8-bit quantization to optimize the model for efficient inference on resource-constrained devices.



