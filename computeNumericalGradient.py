import numpy as np

def computeNumericalGradient(J, theta1,theta2):
    numgrad_w1 = np.zeros(theta1.shape)
    numgrad_w2 = np.zeros(theta2.shape)
    e = 1e-4

    loss1,_,_ = (theta1 - e,theta2 - e)
    loss2,_,_ = (theta1 + e,theta2 + e)





    return numgrad