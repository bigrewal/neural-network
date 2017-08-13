### Neural Network - Computing XOR 

##### Neural Network Architecture

    
    info :: A = ((X*W1)+B1)
            B = ((Z*W2)+B2)
    

    X -------                    | Z-------------
              |                  |              |
    W1 ------ --- A -->Tan(A)--->  W2------------ B --> sigmoid(B) --> Y
              |                  |              |
    B1 ------                    | B2------------
