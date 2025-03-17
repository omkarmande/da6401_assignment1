import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb

global X, y, X_train, y_train, X_val, y_val, X_test, y_test
global Y, y_train_onehot, y_test_onehot
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

split_index = int(0.9 * X_train.shape[0])
X, X_val = X_train[:split_index], X_train[split_index:]
y, y_val = y_train_onehot[:split_index], y_train_onehot[split_index:]


def start_here(args):
    wandb.login()
    if args.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    split_index = int(0.9 * X_train.shape[0])
    X, X_val = X_train[:split_index], X_train[split_index:]
    y, y_val = y_train_onehot[:split_index], y_train_onehot[split_index:]

    model = Model(il_neuron=784, hl_neuron=args.hidden_size, hl_count=args.num_layers, ol_neuron=10, opt=args.optimizer, lr=args.learning_rate, batch=args.batch_size, init=args.weight_init, act=args.activation, loss=args.loss, decay=args.weight_decay)
    model.train(epochs=args.epochs)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    x = np.clip(x, -500, 500)
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def squared_error_loss(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

def accuracy(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)

def initialize_weights(shape, method="xavier"):
    if method == "random":
        return np.random.randn(*shape) * 0.01
    elif method == "xavier":
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    else:
        raise ValueError("Unknown initialization method: Choose 'random' or 'xavier'")

def clip_gradients(grads, clip_value=5.0):
    return [np.clip(g, -clip_value, clip_value) for g in grads]

class Model:
  def get_activation_functions(self, activation_type):
        activations = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "ReLu": (relu, relu_derivative),
            "identity": (identity, identity_derivative)
        }
        return activations.get(activation_type, (sigmoid, sigmoid_derivative))

  def __init__(self, il_neuron, hl_neuron, hl_count, ol_neuron, opt="adam", lr=0.1, batch=4, init="xavier", act="tanh", loss="cross_entropy", decay=0):
    self.layers = [il_neuron] + [hl_neuron]*hl_count + [ol_neuron]
    self.weights = []
    self.biases = []
    self.opt = opt
    self.lr = lr
    self.batch = batch
    self.init = init
    self.act = act
    self.loss = loss
    self.decay = decay
    self.momentum = 0.9
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.epsilon = 1e-6
    self.t = 0

    self.velocities = []
    self.velocities_b = []
    self.squared_grads = []
    self.squared_grads_b = []
    self.m_t_w = []
    self.m_t_b = []
    self.v_t_w = []
    self.v_t_b = []

    self.activation_func, self.activation_derivative = self.get_activation_functions(act)
    self.loss_func = cross_entropy_loss if loss == "cross_entropy" else squared_error_loss

    #initializing and giving shape
    for i in range(len(self.layers) - 1):
        weight_matrix = initialize_weights((self.layers[i], self.layers[i + 1]), method=self.init)
        bias_vector = np.zeros((1, self.layers[i + 1]))
        self.weights.append(weight_matrix)
        self.biases.append(bias_vector)
        self.velocities.append(np.zeros_like(weight_matrix))
        self.velocities_b.append(np.zeros_like(bias_vector))
        self.squared_grads.append(np.zeros_like(weight_matrix))
        self.squared_grads_b.append(np.zeros_like(bias_vector))
        self.m_t_w.append(np.zeros_like(weight_matrix))
        self.m_t_b.append(np.zeros_like(bias_vector))
        self.v_t_w.append(np.zeros_like(weight_matrix))
        self.v_t_b.append(np.zeros_like(bias_vector))

  def feedForward(self, X):
    activations = [X]
    for i in range(len(self.weights) - 1):
      z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
      #print(z.shape)
      a = self.activation_func(z)
      activations.append(a)

    z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
    a_output = softmax(z_output)
    activations.append(a_output)

    return activations

  def backProp(self, X_batch, y_batch, activations):
    batch_size = X_batch.shape[0]
    grads_w = [np.zeros_like(w) for w in self.weights]
    grads_b = [np.zeros_like(b) for b in self.biases]

    if self.loss == "cross_entropy":
        dz = activations[-1] - y_batch
    else:
        dz = (activations[-1] - y_batch) * 2 / batch_size
    grads_w[-1] = np.dot(activations[-2].T, dz) / batch_size + (self.decay * self.weights[-1])
    grads_b[-1] = np.sum(dz, axis=0, keepdims=True) / batch_size

    for i in range(len(self.weights) - 2, -1, -1):
        dz = np.dot(dz, self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
        grads_w[i] = np.dot(activations[i].T, dz) / batch_size + (self.decay * self.weights[i])
        grads_b[i] = np.sum(dz, axis=0, keepdims=True) / batch_size

    self.update_weights(grads_w, grads_b)

  def update_weights(self, grads_w, grads_b):
    grads_w = clip_gradients(grads_w)
    grads_b = clip_gradients(grads_b)

    if self.opt == "sgd":
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * (grads_w[i] + self.decay * self.weights[i])
            self.biases[i] -= self.lr * grads_b[i]

    elif self.opt == "momentum":
        for i in range(len(self.weights)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * (grads_w[i] + self.decay * self.weights[i])
            self.weights[i] += self.velocities[i]

            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.lr * grads_b[i]
            self.biases[i] += self.velocities_b[i]

    elif self.opt == "nesterov":
        for i in range(len(self.weights)):
            lookahead_w = self.weights[i] - self.momentum * self.velocities[i]
            lookahead_b = self.biases[i] - self.momentum * self.velocities_b[i]
            grad_w_lookahead = grads_w[i]
            grad_b_lookahead = grads_b[i]
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * (grad_w_lookahead + self.decay * lookahead_w)
            self.weights[i] += self.velocities[i]
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.lr * grad_b_lookahead
            self.biases[i] += self.velocities_b[i]

    elif self.opt == "rmsprop":
        for i in range(len(self.weights)):
            self.squared_grads[i] = self.beta2 * self.squared_grads[i] + (1 - self.beta2) * grads_w[i]**2
            self.weights[i] -= self.lr * (grads_w[i] + self.decay * self.weights[i]) / (np.sqrt(self.squared_grads[i]) + self.epsilon)
            self.squared_grads_b[i] = self.beta2 * self.squared_grads_b[i] + (1 - self.beta2) * grads_b[i]**2
            self.biases[i] -= self.lr * grads_b[i] / (np.sqrt(self.squared_grads_b[i]) + self.epsilon)

    elif self.opt == "adam":
        self.t += 1
        for i in range(len(self.weights)):
            self.m_t_w[i] = self.beta1 * self.m_t_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_t_w[i] = self.beta2 * self.v_t_w[i] + (1 - self.beta2) * (grads_w[i]**2)
            self.m_t_b[i] = self.beta1 * self.m_t_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_t_b[i] = self.beta2 * self.v_t_b[i] + (1 - self.beta2) * (grads_b[i]**2)
            m_hat_w = self.m_t_w[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_t_w[i] / (1 - self.beta2**self.t)
            m_hat_b = self.m_t_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_t_b[i] / (1 - self.beta2**self.t)
            self.weights[i] -= self.lr * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + self.decay * self.weights[i])
            self.biases[i] -= self.lr * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

  def train(self, epochs=10):
    for epoch in range(epochs):
        for i in range(0, X.shape[0], self.batch):
            X_batch = X[i:i + self.batch]
            y_batch = y[i:i + self.batch]
            activations = self.feedForward(X_batch)
            self.backProp(X_batch, y_batch, activations)

        y_train_pred = self.feedForward(X)[-1]
        train_loss = self.loss_func(y, y_train_pred)
        train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y, axis=1))

        y_val_pred = self.feedForward(X_val)[-1]
        val_loss = self.loss_func(y_val, y_val_pred)
        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))

        wandb.log({"Epoch": epoch+1, "Train Loss": train_loss, "Train Accuracy": train_accuracy*100,
                    "Validation Loss": val_loss, "Validation Accuracy": val_accuracy*100})

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy*100:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy*100:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we", default="cs24m028")
    parser.add_argument("--wandb_project", "-wp", default="Trial")
    parser.add_argument("--dataset", "-d", choices=["mnist","fashion_mnist"])
    parser.add_argument("--epochs","-e", type= int, default=10)
    parser.add_argument("--batch_size","-b", type =int, default=32)
    parser.add_argument("--optimizer","-o", default= "adam", choices=["sgd","momentum","nesterov","rmsprop","adam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
    parser.add_argument("--momentum","-m", default=0.9,type=float)
    parser.add_argument("--beta","-beta", default=0.9, type=float)
    parser.add_argument("--beta1","-beta1", default=0.9,type=float)
    parser.add_argument("--beta2","-beta2", default=0.999,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 0.000001)
    parser.add_argument("--weight_decay","-w_d", default=0,type=float)
    parser.add_argument("-w_i","--weight_init", default="xavier",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=4)
    parser.add_argument("--hidden_size","-sz",type=int, default=128)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","ReLu"], default="tanh")

    args = parser.parse_args()
    print(args.epochs)
    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_entity)
    start_here(args)
    wandb.finish()


