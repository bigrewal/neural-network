import scipy.io as sci
import numpy as np

def loadData(file):
    dict = sci.loadmat(file)
    x = dict['X']
    y = dict['y']

    return x,y

def process_data(x,y,m,n,num_labels):

    X = np.ones((m, n + 1))
    X[:, 1:n + 1] = x

    I = np.identity(num_labels)
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, :] = I[y[i] - 1, :]

    return X,Y

def loadWeights(file):
    dict = sci.loadmat(file)
    w1 = dict['Theta1']
    w2 = dict['Theta2']

    return w1,w2