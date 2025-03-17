# da6401_assignment1
Link for my report - https://wandb.ai/omkarmande-iit-madras/assignment_01/reports/DA6401-Assignment-1--VmlldzoxMTgzMzM1MQ?accessToken=zy4ogvkiqwvyzuf5wldsmzgrddcplxfloekobzrc5q50pi7mdol3ppazqd0wiiqq
GitHub Link - https://github.com/omkarmande/da6401_assignment1

Neural Network Implementation for MNIST and Fashion-MNIST Classification

Overview - 
This repository contains a Python implementation of a fully connected neural network designed to classify images from the MNIST and Fashion-MNIST datasets. The neural network is built from scratch using NumPy, and it supports various configurations such as different activation functions, optimization algorithms, and loss functions. The project also integrates with Weights & Biases (W&B) for experiment tracking and visualization.


Features
Datasets: Supports both MNIST and Fashion-MNIST datasets.

Activation Functions: Implements Sigmoid, Tanh, ReLU, and Identity activation functions.

Optimizers: Includes SGD, Momentum, Nesterov Accelerated Gradient, RMSProp, and Adam optimizers.

Loss Functions: Supports Cross-Entropy and Mean Squared Error (MSE) loss functions.

Weight Initialization: Provides options for Random and Xavier weight initialization.

Regularization: Includes L2 regularization (weight decay) to prevent overfitting.

Experiment Tracking: Integrates with Weights & Biases to log training and validation metrics.

Installation
Clone the Repository:
git clone https://github.com/your-username/neural-network-from-scratch.git
cd neural-network-from-scratch

Install Dependencies:
Ensure you have Python 3.x installed. Then, install the required packages:
pip install numpy matplotlib scikit-learn keras wandb

Usage
Running the Script
To train the neural network, use the following command:

python main.py --dataset mnist --epochs 10 --batch_size 32 --optimizer adam --learning_rate 0.0001 --hidden_size 128 --num_layers 4 --activation tanh --loss cross_entropy
Command Line Arguments
--dataset (-d): Choose between mnist or fashion_mnist.

--epochs (-e): Number of training epochs (default: 10).

--batch_size (-b): Batch size for training (default: 32).

--optimizer (-o): Optimization algorithm (sgd, momentum, nesterov, rmsprop, adam).

--loss (-l): Loss function (mean_squared_error, cross_entropy).

--learning_rate (-lr): Learning rate (default: 0.0001).

--hidden_size (-sz): Number of neurons in each hidden layer (default: 128).

--num_layers (-nhl): Number of hidden layers (default: 4).

--activation (-a): Activation function (identity, sigmoid, tanh, ReLu).

--weight_init (-w_i): Weight initialization method (random, xavier).

--weight_decay (-w_d): L2 regularization strength (default: 0).

--wandb_entity (-we): Your Weights & Biases entity name (default: cs24m028).

--wandb_project (-wp): Weights & Biases project name (default: Trial).

Example Commands
Train on MNIST with Adam Optimizer:

python main.py --dataset mnist --epochs 20 --batch_size 64 --optimizer adam --learning_rate 0.001 --hidden_size 256 --num_layers 3 --activation ReLu --loss cross_entropy

Train on Fashion-MNIST with SGD and Momentum:

python main.py --dataset fashion_mnist --epochs 15 --batch_size 128 --optimizer momentum --learning_rate 0.01 --hidden_size 512 --num_layers 5 --activation tanh --loss mean_squared_error


Code Structure
Model Initialization: The Model class initializes the neural network with the specified architecture, activation functions, and optimization algorithm.

Forward Propagation: The feedForward method computes the activations for each layer.

Backpropagation: The backProp method calculates gradients and updates weights using the chosen optimizer.

Training Loop: The train method iterates over the dataset, performs forward and backward passes, and logs metrics using W&B.

Weights & Biases Integration
The script logs the following metrics to W&B:

Train Loss: Loss on the training set.

Train Accuracy: Accuracy on the training set.

Validation Loss: Loss on the validation set.

Validation Accuracy: Accuracy on the validation set.

To view the logged metrics, ensure you are logged into your W&B account and navigate to the project dashboard.

Customization
You can easily customize the neural network by modifying the command-line arguments or directly editing the code. For example:

Add new activation functions or loss functions.

Experiment with different weight initialization methods.

Tune hyperparameters like learning rate, batch size, and number of layers.
