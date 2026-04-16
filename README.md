# Mini Deep Learning Framework

This project is a small deep learning framework built from scratch using NumPy.  
The goal is to better understand how neural networks, backpropagation, and training loops work internally.


## Features:

- custom neural network implementation
- Dense, ReLU, Softmax layers
- Backpropagation from scratch
- Training loop with batching
- Model saving/loading
- Evaluation metrics (accuracy, loss, Confusion Matrix)

## Project Structure:

- core/ # neural network layers and model logic
- training/ # training loop and optimizer
- metrics/ # evaluation functions
- tests/ # debugging and overfit tests
- visualisation/ # plots and analysis
- data/ # MNIST dataset (not included)
- train.py # entry point for training

## Setup

1. Clone the repository
2. Download the MNIST dataset manually
3. Place it inside the `data/` folder

Example structure:
- data/mnist_train.csv
- data/mnist_test.csv

## Run Training

python train.py