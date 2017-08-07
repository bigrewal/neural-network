import numpy as np
from sigmoid import sigmoid_grad
from feedForward import forward

def loss(X, Y, w1, w2, lmbda):
    m, _ = X.shape
    A3, cache = forward(X, w1, w2)

    Z2, sig_out, A2, Z3 = cache

    J = 1 / m * np.sum(np.sum((-Y * np.log(A3) - (1 - Y) * np.log(1-A3))))

    s1 = np.sum(np.sum(np.power(w1[:, 1:w1.shape[1]], 2)))
    s2 = np.sum(np.sum(np.power(w2[:, 1:w2.shape[1]], 2)))
    reg = lmbda / (2 * m) * (s1+s2)

    loss = J + reg

    #  (5000,401)                           (5000,26)
    # X -------       (5000,25)       A2-------------    (5000,10)
    #    X*W1   |---Z2-->SIGMOID(Z2)---> |            -----> Z3 --> SIGMOID(Z3)-->A3
    # W1 ------                        W2-------------
    # (401,25)                               (26,10)                             (5000,10)

    delta3 = A3 - Y
    delta2 = (delta3.dot(np.transpose(w2)) * sigmoid_grad(A2))[:,1:]

    Delta2 = np.transpose(A2).dot(delta3)
    Delta1 = np.transpose(X).dot(delta2)

    theta_new = np.zeros(w1.shape)
    theta_new[1:,:] = w1[1:,:]
    dw1 = 1/m * Delta1 + lmbda/m * theta_new

    theta_new = np.zeros(w2.shape)
    theta_new[1:, :] = w2[1:, :]
    dw2 = 1/m * Delta2 + lmbda/m * theta_new

    return loss,dw1,dw2

