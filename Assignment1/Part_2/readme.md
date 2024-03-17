# Multi-Layer Perceptron (MLP) Implementation in NumPy

This repository contains an <span style="color:red;"> incomplete </span> implementation of a Multi-Layer Perceptron (MLP), a type of feedforward artificial neural network, using only NumPy for computations. This code is designed for educational purposes to help you guys understand the basics of neural networks, including forward propagation, backpropagation, and the training process.

## Overview

The MLP is implemented across three Python files:

- `mlp_numpy.py`: Defines the `MLP` class, which sets up the neural network architecture including the initialization of layers, forward pass, backward pass (for backpropagation), and a step function for updating weights using gradient descent.

- `modules.py`: Contains the implementation of various components used in the MLP such as linear (fully connected) layers, the ReLU activation function, softmax function, and the cross-entropy loss. These components are modular, making the code easier to understand and modify.

- `train_mlp_numpy.py`: A script that demonstrates how to use the `MLP` class for training on a simple dataset (generated using sklearn's `make_moons` function). It includes functions for data loading, model training, accuracy computation, and one-hot encoding of labels.

## Requirements

- Python 3.x
- NumPy
- scikit-learn

## Training the MLP

The training process involves several steps, including data preparation, model initialization, and iterative training over multiple epochs. The `train_mlp_numpy.py` script you finished should encapsulates this process, showing us you how to:

1. Generate a dataset.
2. Split the dataset into training and testing sets.
3. Encode the labels in one-hot format.
4. Initialize the MLP model with a specified architecture (e.g., number of hidden layers and units per layer).
5. Train the model using gradient descent, periodically evaluating its performance on the test set.

### Running the Training Script

To train the MLP model, run the `train_mlp_numpy.py` script from the command line. You can customize the training process by specifying command-line arguments such as the number of hidden units, learning rate, number of epochs, and evaluation frequency.

Example command:

```bash
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 0.01 --max_steps 1500 --eval_freq 10
