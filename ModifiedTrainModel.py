import numpy as np
import pvml
import matplotlib.pyplot as plt
import sys

# COMPUTE THE ACCURACY
def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100


# MinMax Normalization
def minmax_normalization(Xtrain, Xtest):
    xmin = Xtrain.min(0)
    xmax = Xtrain.max(0)
    Xtrain = (Xtrain - xmin) / (xmax - xmin)
    Xtest = (Xtest - xmin) / (xmax - xmin)
    return Xtrain, Xtest


# Mean-Variance Normalization
def meanvar_normalization(Xtrain, Xtest):
    mu = Xtrain.mean(0)
    std = Xtrain.std(0)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    return Xtrain, Xtest


# MAX-Absolute Normalization
def maxabs_normalization(Xtrain, Xtest):
    amax = np.abs(Xtrain).max(0)
    Xtrain = Xtrain / amax
    Xtest = Xtest / amax
    return Xtrain, Xtest


if __name__ == "__main__":
    # LOAD THE LIST OF CLASSES
    words = open("classes.txt").read().split()
    print(words)

    # LOAD THE TRAINING AND TEST DATA
    data = np.load("train.npz")
    Xtrain = data["arr_0"]
    Ytrain = data["arr_1"]
    print(Xtrain.shape, Ytrain.shape)
    data = np.load("test.npz")
    Xtest = data["arr_0"]
    Ytest = data["arr_1"]
    print(Xtest.shape, Ytest.shape)
    np.set_printoptions(threshold=sys.maxsize)
    print(Xtrain[0:20])
    # MEAN/VARIANCE NORMALIZATION
    Xtrain, Xtest = meanvar_normalization(Xtrain, Xtest)

    # CREATE AND TRAIN THE MULTI-LAYER PERCEPTRON
    net = pvml.MLP([1600, 35])
    m = Ytrain.size
    plt.ion()
    train_accs = []
    test_accs = []
    epochs = []
    batch_size = 10
    for epoch in range(10):
        net.train(Xtrain, Ytrain, 1e-4, steps=m // batch_size,
                  batch=batch_size)
        if epoch % 5 == 0:  # to speed up
            train_acc = accuracy(net, Xtrain, Ytrain)
            test_acc = accuracy(net, Xtest, Ytest)
            print(epoch, train_acc, test_acc)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)
            plt.clf()
    plt.plot(epochs, train_accs)
    plt.plot(epochs, test_accs)
    plt.xlabel("epochs")
    plt.ylabel("accuracies (%)")
    plt.legend(["train", "test"])
    plt.pause(0.01)
    plt.ioff()
    plt.show()

    # SAVE THE MODEL TO DISK
    net.save("mlp2.npz")
