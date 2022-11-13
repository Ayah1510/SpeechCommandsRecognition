import numpy as np
import pvml
import matplotlib.pyplot as plt


# MEASURE THE ACCURACY ON A GIVEN SET
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


# DISPLAY THE CONFUSION MATRIX
def show_confusion_matrix(Y, predictions):
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3)
    plt.clf()
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Blues)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")


if __name__ == "__main__":

    # LOAD THE LIST OF CLASSES
    words = open("classes.txt").read().split()
    print(words)

    # LOAD TRAINING AND TEST DATA
    data = np.load("train.npz")
    Xtrain = data["arr_0"]
    Ytrain = data["arr_1"]
    print(Xtrain.shape, Ytrain.shape)
    print(Xtrain)
    data = np.load("test.npz")
    Xtest = data["arr_0"]
    Ytest = data["arr_1"]
    print(Xtest.shape, Ytest.shape)

    # MEAN/VARIANCE NORMALIZAZION
    Xtrain, Xtest = meanvar_normalization(Xtrain, Xtest)

    # LOAD THE WEIGHTS
    net = pvml.MLP.load("mlp2.npz")

    # SHOW THE WEIGHTS FOR A GIVEN CLASS
    w = net.weights[0]
    image = w[:, 3].reshape(20, 80)
    plt.imshow(image, cmap="seismic",vmin=-0.6, vmax=0.6)
    plt.title(words[3])
    plt.colorbar()
    plt.show()

    predictions, probs = net.inference(Xtest)
    print(probs[30])
    show_confusion_matrix(Ytest, predictions)
    plt.show()
    # COMPUTE AND PRINT THE CONFISION MATRIX
    nclasses = len(words)
    cm = np.zeros((nclasses, nclasses))
    for i in range(Ytest.size):
        cm[Ytest[i], predictions[i]] += 1
    cm = cm / cm.sum(1, keepdims=True)
    for i in range(nclasses):
        for j in range(nclasses):
            print(f"{cm[i, j]:.2f}", end=" ")
        print()

    #to print the confusion matrix as counters
    #i=0
    #j=0
    #cm = np.zeros((nclasses, nclasses))
    #for i in range(Ytest.size):
    #    cm[Ytest[i], predictions[i]] += 1
    #for i in range(nclasses):
    #    for j in range(nclasses):
    #        print(int(cm[i, j]), end=" ")
    #    print()
