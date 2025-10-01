import numpy as np

x1 = 1
x2 = 0

x = np.array([x1, x2])

# weights and biases from training
w1 = np.array([
    [-1.14982244, -1.14982243],
    [-1.6631927 , -1.6631927 ]
])
b1 = np.array([
    [2.29948808, 1.66131385]
])
w2 = np.array([
    [ 2.58761599, -2.38728327]
])
b2 = np.array([
    [-1.97479652]
])

def leakyrelu(z, alpha=0.01): # activation function
    return np.where(z > 0, z, alpha * z)

z1 = x @ w1.T + b1 # layer 1
h = leakyrelu(z1)

z2 = h @ w2.T + b2 # layer 2
ypred = leakyrelu(z2) # output

print("Final predicted value for inputs x1:", x1, "and x2:", x2, "is \n", ypred)
