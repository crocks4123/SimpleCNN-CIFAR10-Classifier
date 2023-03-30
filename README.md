# Convolutional Neural Network for CIFAR-10 Classification
This repository contains code for training a Convolutional Neural Network (CNN) for the CIFAR-10 image classification task. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Dependencies
The code is written in Python and uses the following libraries:

- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
To install the required libraries, run the following command:
` pip install tensorflow numpy matplotlib ` 
## Usage
The code is provided in a Python script cifar10_cnn.py. You can run the script using the following command:
python cifar10_cnn.py
The script will automatically download the CIFAR-10 dataset and train a CNN on the dataset. You can visit this site for the details and specifications of the data set.
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

## Model Architecture
The CNN architecture used in this code consists of three convolutional layers followed by two fully connected layers. The architecture is as follows:
| Layer (type)            | Output Shape        | Param #   |
|------------------------|---------------------|-----------|
| conv2d                 | (None, 30, 30, 32)   | 896       |
| max_pooling2d          | (None, 15, 15, 32)   | 0         |
| conv2d_1               | (None, 13, 13, 64)   | 18496     |
| max_pooling2d_1        | (None, 6, 6, 64)     | 0         |
| conv2d_2               | (None, 4, 4, 64)     | 36928     |
| flatten                | (None, 1024)         | 0         |
| dense                  | (None, 64)           | 65600     |
| dense_1                | (None, 10)           | 650       |

|Total params: 122,570    |
|Trainable params: 122,570|
|Non-trainable params: 0  |

## Results
The model is trained for 10 epochs on the CIFAR-10 dataset. The training and validation accuracy and loss are plotted using Matplotlib.

## Credits
The code is adapted from the official TensorFlow tutorial on CIFAR-10 classification.
