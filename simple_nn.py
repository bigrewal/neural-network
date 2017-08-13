import numpy as np

#Hyperparameters
n_hidden = 10
n_in = 10
n_out = 10
n_samples = 300
tot_ittrs = 200

learning_rate = 0.01

# Initialise Weights
W1 = np.random.normal(scale=0.5, size=(n_in, n_hidden))  #(10,10)
W2 = np.random.normal(scale=0.5, size=(n_hidden, n_out)) #(10,10)

# Initialise Biases
b1 = np.zeros(n_hidden)
b2 = np.zeros(n_out)

# Generate our training data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
Y = X ^ 1

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def train(X, y, W1, W2, B1, B2):

    # forward
    A = np.dot(X, W1) + B1
    # Z = np.tanh(A)
    Z = sigmoid(A)

    B = np.dot(Z, W2) + B2
    Y = sigmoid(B)

    # Backward propogation
    delta3 = Y - y                                          # Error
    d_b2 = np.sum(delta3,axis=0)                            # derivate of error with respect to b2
    d_w2 = np.dot(np.transpose(Z),delta3)                   # derivate of error with respect to W2
    d_Z = np.dot(delta3,W2)                                 # derivate of error with respect to Z

    # d_tan = tanh_grad(A) * d_Z
    d_sig = sigmoid_grad(A) * d_Z
    d_b1 = np.sum(d_sig,axis=0)                              #derivate of error with respect to b1
    d_w1 = np.dot(np.transpose(X),d_sig)                     #derivate of error with respect to W1

    loss = -np.mean(y * np.log(Y) + (1 - y) * np.log(1 - Y))

    return  loss, [d_w1,d_w2,d_b1,d_b2]

def predict(X,W1, W2, B1, B2):
    A = np.dot(X, W1) + B1
    Z = sigmoid(A)

    B = np.dot(Z, W2) + B2
    Y = sigmoid(B)
    return (Y > 0.5).astype(int)

# Train
for i in range(tot_ittrs):
    loss, grads = train(X, Y, W1,W2,b1,b2)
    W1 = W1 - (learning_rate * grads[0])
    W2 = W2 - (learning_rate * grads[1])
    b1 = b1 - (learning_rate * grads[2])
    b2 = b2 - (learning_rate * grads[3])

    print("Itteration: %d, Loss: %.8f" % (i, loss))

pred = predict(X,W1, W2, b1, b2)
print(pred == Y)
accuracy = np.mean(pred == Y)
print("Accuracy on the training set: ",accuracy)

# Let's Predict Something..
x = np.random.binomial(1, 0.5, n_in)
x = np.reshape(x,(1,10))
print(x)
print("XOR prediction: ",predict(x,W1, W2, b1, b2))