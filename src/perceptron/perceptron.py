import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
    

def predict(X, W, b):
    A = model(X, W, b)
    print(A)
    return A >= 0.5


def log_loss(A, y):
    m = len(y)
    epsillon = 1e-15
    return 1 / m * np.sum(-y * np.log(A + epsillon) - (1 - y) * np.log(1 - A + epsillon))


def gradients(X, y, A):
    m = len(y)
    dW = 1 / m * np.dot(X.T, A - y)
    db = 1 / m * np.sum(A - y)
    return dW, db


def update(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b -learning_rate * db
    return (W, b)


def artificial_neurone(X, y, learning_rate = 0.1, n_iterations = 100, mod = False):
    W , b = initialisation(X)
    loss = []
    acc = []
    for i in tqdm(range(n_iterations)):
        A = model(X, W, b)
        if mod:
            if i % 10 == 0 :
                loss.append(log_loss(A, y))
                y_pred = predict(X, W, b)
                acc.append(accuracy_score(y, y_pred))
        else:
            loss.append(log_loss(A, y))
            y_pred = predict(X, W, b)
        dW, db = gradients(X, y, A)
        W, b = update(W, b, dW, db, learning_rate)
    y_pred = predict(X, W, b)
    print(f'la performance du modèle est : {accuracy_score(y, y_pred)}')
    return W, b, loss, acc
    

