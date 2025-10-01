import numpy as np

a = 0.08  #learning rate
epoch = 2200

def leakyrelu(z, alpha=0.5): # activation function
    return np.where(z > 0, z, alpha * z)

def leakyrelu_derivative(z, alpha=0.5):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz


x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([[0],[1],[1],[0]])

n = len(x) # batch size

rng = np.random.default_rng(0)
w1 = rng.uniform(-1, 1, (2,2)) # weights and biases
b1 = rng.uniform(-1, 1, (1,2))
w2 = rng.uniform(-1, 1, (1,2))
b2 = rng.uniform(-1, 1, (1,1))


for epoch in range(epoch):
    
    z1 = x @ w1.T + b1 # layer 1
    h = leakyrelu(z1)

    z2 = h @ w2.T + b2 # layer 2
    ypred = leakyrelu(z2) # the predicted outputs

    L = ((y - ypred) ** 2) / n # loss function

    # start of the chain rule
    dL = 2 * (ypred - y) / n # derivative of loss function

    dypred = leakyrelu_derivative(z2) * dL # derivative of prediction

    dz2_h = w2 * dypred # derivative of layer 2
    dz2_w2 = (dypred.T @ h)
    dz2_b2 = np.sum(dypred, axis=0)

    dh = leakyrelu_derivative(z1) * dz2_h # derivative of layer 1
    dz1_w1 = (dh.T @ x)
    dz1_b1 = np.sum(dh, axis=0) 
    # end of the chain rule

    w2 -= a * dz2_w2 # reassign new weights and biases
    b2 -= a * dz2_b2
    w1 -= a * dz1_w1
    b1 -= a * dz1_b1

    print("The predicted value for epoch ", epoch, "is \n", ypred) # print the prediction for every epoch

print("\nFinal predicted value is \n", ypred)
print("\nFinal weights for layer 1 are \n", w1)
print("\nFinal biases for layer 1 are \n", b1)
print("\nFinal weights for layer 2 are \n", w2)
print("\nFinal biases for layer 2 are \n", b2)
