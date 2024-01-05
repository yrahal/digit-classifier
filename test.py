import gzip
import os
import pickle
import numpy as np

from network import NeuralNetwork


def load_mnist():
    data_file = gzip.open(os.path.join(os.getenv("P4D_INPUT"), "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [vectorized_result(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = val_data[1]
    val_data = list(zip(val_inputs, val_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    return train_data, val_data, test_data


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


if __name__ == "__main__":
    np.random.seed(42)

    layers = [784, 30, 10]
    learning_rate = 0.01
    mini_batch_size = 16
    epochs = 100

    # Initialize train, val and test data
    train_data, val_data, test_data = load_mnist()

    nn = NeuralNetwork(layers, learning_rate, mini_batch_size, "relu")
    
    with open(os.path.join(os.getenv("P4D_INPUT"),'model.pkl'), 'rb') as handle:
        o = pickle.load(handle)
        nn.weights = o["w"]
        nn.biases = o["b"]

    accuracy = nn.validate(test_data) / 100.0
    print(f"Test Accuracy: {accuracy}%.")
