import numpy as np
#sizes = [2,4,1]
#ysizes = sizes[1:]
#xsizes = sizes[:-1]
#
#biases = [np.random.randn(y,1) for y in sizes[1:]]
#weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

x1 = np.ones((500,1))
x2 = np.zeros((500,1))
x = np.concatenate((x1,x2))
w = np.random.randn(1000,1)/np.sqrt(500)
b = np.random.randn(1,1)
z = w.T @ x + b

std = np.std(w)
var = np.std(w)
mean = np.mean(w)