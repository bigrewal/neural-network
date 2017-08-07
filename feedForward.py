from sigmoid import sigmoid
import numpy as np

def forward(X,w1,w2):
    #Input --> Hidden
    Z2 = X.dot(w1)
    sig_out = sigmoid(Z2)
    m,n = sig_out.shape

    A2 = np.ones((m, n + 1))
    A2[:, 1:n + 1] = sig_out

    # Hidden --> Output
    Z3 = A2.dot(w2)
    A3 = sigmoid(Z3)

    cache = Z2,sig_out,A2,Z3
    return A3,cache