### Neural Network - Computing XOR 

##### Neural Network Architecture

    
    A = ((X*W1)+B1)
    B = ((Z*W2)+B2)

      (300,10)                              (300,10)
    X -------                           | Z-------------
      (10,10) |       (300,10)          |    (10,10)   |        (300,10)
    W1 ------ --- A -->sigmoid(A)--->  W2------------  B --> sigmoid(B) --> Y
     (10,)     |                         |    (10,)    |
    B1 ------                           | B2------------

