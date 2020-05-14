import numpy as np
np.random.seed(1) 

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
X, Y = load_planar_dataset()
#print(X.shape)
#print(Y.shape)

#X = np.array([
#        [99,1,1,1,1,1,1,1,1,1],
#        [1,99,1,1,1,1,1,1,1,1],
#        [1,1,99,1,1,1,1,1,1,1],
#        [1,1,1,99,1,1,1,1,1,1],
#        [1,1,1,1,99,1,1,1,1,1],
#        [1,1,1,1,1,99,1,1,1,1],
#        [1,1,1,1,1,1,99,1,1,1],
#        [1,1,1,1,1,1,1,99,1,1],
#        [1,1,1,1,1,1,1,1,99,1],
#        [1,1,1,1,1,1,1,1,1,99]
#        ]).T
#
#Y = np.array([
#        [0,0,0,0],
#        [0,0,0,1],
#        [0,0,1,0],
#        [0,0,1,1],
#        [0,1,0,0],
#        [0,1,0,1],
#        [0,1,1,0],
#        [0,1,1,1],
#        [1,0,0,0],
#        [1,0,0,1]
#        ]).T

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#    return np.exp(z) / (np.exp(z)+1)**2

def define_NN_structure(X,Y):
    n_x = X.shape[0] #input neurons
    n_h = 4 #hidden neurons
    n_y = Y.shape[0] #output neurons
    return n_x, n_h, n_y
    
def initialize_params(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01 #n[L] by n[L-1]
    B1 = np.zeros((n_h,1)) #n[L] by 1 
    W2 = np.random.randn(n_y,n_h)*0.01 #n[L] by n[L-1]
    B2 = np.zeros((n_y,1)) #n[L] by 1 
    return W1, B1, W2, B2
    
def forward_prop(W1, B1, W2, B2):
    Z1 = W1 @ X + B1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + B2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def compute_cost(A2, Y):
    m = Y.shape[1]
    L = 1/m * np.sum((A2 - Y)**2) #MSE
#    L = -1/m * np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) #CEL
    return L
    
def backward_prop(X, Y, Z1, A1, Z2, A2):
    m = Y.shape[1]
#    dA2 = 1/m * 2*(A2 - Y) #1,400 #MSE
    dA2 = -1/m * (Y/A2 - (1-Y)/(1-A2)) #it's funny using this for differentiation works better but using this for actual loss function doesn't seem to be working any better
#    dA2 = 1/m * (A2 - Y)
    dZ2 = sigmoid_prime(Z2) #1,400
    dW2 = A1 #4,400
    dB2 = np.ones((1,A1.shape[1]))#1,400
    dA1 = W2 #1,4
    dZ1 = sigmoid_prime(Z1) #4,400
    dW1 = X #2,400
    dB1 = np.ones((1,X.shape[1])) #1,400

    delta2 = dA2 * dZ2 #1,400 # first operation = scalar multiplication that results in a vector we refer to as the "backpropagating error, delta 2"
    dLdW2 = delta2 @ dW2.T #1,4 # each error delta2 needs to be multiplied by each previous activation
    dLdB2 = delta2 @ dB2.T #1,1

    delta1 = delta2.T @ dA1 * dZ1.T #400,4
    dLdW1 = delta1.T @ dW1.T #4,2
    dLdB1 = delta1.T @ dB1.T #4,1

    return dLdW2, dLdB2, dLdW1, dLdB1, dA2, dZ2, dW2, dB2, dA1, dZ1, dW1, dB1
    
def update_params(W1, B1, W2, B2, dLdW2, dLdB2, dLdW1, dLdB1, alpha):
    W1 = W1 - alpha * dLdW1
    B1 = B1 - alpha * dLdB1
    W2 = W2 - alpha * dLdW2
    B2 = B2 - alpha * dLdB2    
    return W1, B1, W2, B2 

n_x, n_h, n_y = define_NN_structure(X,Y)
W1, B1, W2, B2 = initialize_params(n_x, n_h, n_y)

for i in range(10000):
    Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2)
    L = compute_cost(A2, Y)
    if i % 500 == 0:
        print('Loss is: ' + str(L))
    dLdW2, dLdB2, dLdW1, dLdB1, dA2, dZ2, dW2, dB2, dA1, dZ1, dW1, dB1 = backward_prop(X, Y, Z1, A1, Z2, A2)
    W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dLdW2, dLdB2, dLdW1, dLdB1, 2.5)