from load import loadData,process_data,loadWeights
from costFunction import loss
from feedForward import forward
import numpy as np
import matplotlib.pyplot as plt

input_layer = 400
hidden_layer = 25
num_labels = 10
learning_rate = 0.01
lmbda = 0.1;
total_itrs = 200

x,y = loadData("data.mat")
# w1,w2 = loadWeights("weights.mat")
#
# w1 = np.transpose(w1)
# w2 = np.transpose(w2)

m,n = x.shape
X,Y = process_data(x,y,m,n,num_labels=num_labels)
m,n = X.shape

print("=========== Time to Train... ===============")
w1 = np.random.rand(input_layer+1,hidden_layer)
w2 = np.random.rand(hidden_layer+1,num_labels)

loss_history = np.zeros(total_itrs);
for i in range(100):
    cost, dw1, dw2 = loss(X,Y,w1,w2,lmbda)
    print(cost)
    loss_history[i] = cost
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2

plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Itterations')
plt.show()

print("=========== Accuracy on the training set ===============")
pred,_ = forward(X,w1,w2)
print(np.argmax(pred[0]))
print(np.mean((np.argmax(pred,1)+1) == y)*100)