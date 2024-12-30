# Neural Networks

This folder contains an implementation of a simple neural network from scratch.

## Files

- **`model.py`**: Contains the implementation of the neural network model, including forward propagation, backpropagation, and training methods.
- **`train.py`**: Script to train the neural network model on the MNIST dataset, including data loading, training loop, and evaluation.

## Highlights

- This implementation is coded entirely from scratch to understand the workings of a neural network.
- No pre-built libraries (e.g., TensorFlow, PyTorch) were used for the core architecture. We only used NumPy for numerical computations.
- The goal was to deeply understand the workings of a neural network by building it myself.

## How to Use

1. Ensure you have the required dependencies installed (e.g., NumPy, Pandas, Matplotlib).
2. Download the MNIST dataset and place the `mnist_train.csv` and `mnist_test.csv` files in the same directory as the scripts.
3. Run `train.py` to train the model:
   ```bash
   python train.py
Feel free to explore the code to understand the implementation and adapt it to your needs!

## Features Implemented
- Sigmoid activation function and its derivative
- One-hot encoding for labels
- Forward propagation
- Backpropagation
- Gradient descent and stochastic gradient descent
- Mini-batch training
- Accuracy evaluation

## Data Source 
This project uses the MNIST dataset.

### Attribution.
The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.

### Terms of Use
Please see the terms of use for the MNIST dataset: [MNIST Terms of Use](http://yann.lecun.com/exdb/mnist/)