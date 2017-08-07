import numpy as np
from costFunction import loss
from computeNumericalGradient import computeNumericalGradient

def checkGrad(lmbda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = np.random.rand(input_layer_size,hidden_layer_size)
    Theta2 = np.random.rand(hidden_layer_size,num_labels)
    # Reusing debugInitializeWeights to generate X
    X = np.random.rand(m, input_layer_size - 1)
    y = 1 + np.mod(range(m), num_labels).T

    # Short hand for cost function
    def costFunc(p1,p2):
        return loss(X, y, p1, p2, lmbda)

    _, dw1,dw2 = costFunc(Theta1,Theta2)

    numgrad = computeNumericalGradient(costFunc, Theta1,Theta2)