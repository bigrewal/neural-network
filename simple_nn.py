import numpy as np
import time

#Hyperparameters
n_hidden = 10
n_in = 10
n_out = 10
n_samples = 300

learning_rate = 0.01
momentum = 0.9

# Initialise Weights
W1 = np.random.normal(scale=0.5, size=(n_in, n_hidden))  #(10,10)
W2 = np.random.normal(scale=0.5, size=(n_hidden, n_out)) #(10,10)

# Initialise Biases
b1 = np.zeros(n_hidden)
b2 = np.zeros(n_out)

params = [W1,W2,b1,b2]

# Generate our training data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
Y = X ^ 1

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#Gradient for our tanh activation function
def tanh_grad(x):
    return 1 - np.tanh(x)**2

def sigmoid_grad(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def train(x, y, W1, W2, B1, B2):

    # forward
    A = np.dot(x, W1) + B1
    # Z = np.tanh(A)
    Z = sigmoid(A)

    B = np.dot(Z, W2) + B2
    Y = sigmoid(B)

    # Backward propogation
    delta3 = Y - y                  # Error with respect to sigmoid
    d_b2 = delta3                   # Bias 2 gradient
    d_w2 = np.outer(Z,delta3)       # Weight 2 Gradient
    d_Z = np.dot(W2, delta3)

    # d_tan = tanh_grad(A) * d_Z
    d_tan = sigmoid_grad(A) * d_Z
    d_b1 = d_tan                    #Bias 1 gradient
    d_w1 = np.outer(x,d_tan)        #Weight 1 gradient

    loss = -np.mean(y * np.log(Y) + (1 - y) * np.log(1 - Y))

    return  loss, [d_w1,d_w2,d_b1,d_b2]


def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Train
for epoch in range(100):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(X.shape[0]):
        loss, grad = train(X[i], Y[i], *params)
        params[0] = params[0] - (learning_rate * grad[0])
        params[1] = params[1] - (learning_rate * grad[1])
        params[2] = params[2] - (learning_rate * grad[2])
        params[3] = params[3] - (learning_rate * grad[3])

        err.append(loss)

    print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

# Try to predict something
x = np.random.binomial(1, 0.5, n_in)
print("XOR prediction:")
print(x)
print(predict(x, *params))
