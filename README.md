# Emotion Classification from Voice Scalograms using CNNs

This project investigates the use of Convolutional Neural Networks (CNNs) for automatic emotion recognition from human speech. Audio signals are transformed into scalogram images using the Continuous Wavelet Transform (CWT), converting the problem into an image classification task.

The study evaluates three classical CNN architectures:

* LeNet-5
* AlexNet
* VGG16

In addition to the baseline models, several architectural modifications were explored, including:

* Batch Normalization
* Dropout Regularization
* Adam and AdamW Optimizers
* Reduced Fully Connected Layers
* Global Average Pooling (GAP)
* Increased Convolutional Capacity

A total of **15 experiments** were conducted to analyze how each modification affects model performance, convergence behavior, computational cost, and generalization capability.

## Dataset

The dataset contains **30,000 RGB scalograms (224×224 pixels)** generated from speech signals and evenly distributed among five emotional categories:

* Disgust
* Fear
* Happy
* Neutral
* Sad

Dataset split:

* Training: 24,000 images (80%)
* Validation: 3,000 images (10%)
* Test: 3,000 images (10%)

## Objectives

* Compare the performance of classical CNN architectures on emotion recognition.
* Analyze the impact of normalization and regularization techniques.
* Study the relationship between model complexity and classification performance.
* Identify limitations of CNNs when dealing with emotional speech representations.

## Technologies

* Python 3.10
* PyTorch 2.x
* CUDA 12.x
* NumPy
* Scikit-Learn
* Matplotlib
* Google Colab
* NVIDIA A100 GPU

## Experimental Variants

### LeNet-5

* Baseline
* Batch Normalization
* Adam Optimizer
* Wider Convolutional Layers
* Wider Layers + BatchNorm + AdamW

### AlexNet

* Baseline
* Batch Normalization
* Dropout
* Reduced Fully Connected Layers
* BatchNorm + Reduced FC + AdamW

### VGG16

* Baseline
* Batch Normalization
* Global Average Pooling
* BatchNorm + Dropout
* BatchNorm + GAP + AdamW

## Results

The experiments revealed that emotion classification from scalograms is a challenging task.

Key findings include:

* Batch Normalization was the most impactful modification.
* VGG16 failed to learn meaningful representations without BatchNorm.
* AlexNet with BatchNorm, reduced FC layers, and AdamW achieved the best balance between performance and computational cost.
* Reducing fully connected layers significantly decreased parameters without affecting accuracy.
* None of the evaluated models surpassed 24% test accuracy, indicating that the extracted visual features were insufficient for robust emotion discrimination.

## Future Work

Potential improvements include:

* Transfer Learning with ImageNet-pretrained models.
* Data augmentation specialized for scalograms.
* Class-weighted loss functions.
* Residual architectures such as ResNet.
* Advanced learning-rate schedulers.


