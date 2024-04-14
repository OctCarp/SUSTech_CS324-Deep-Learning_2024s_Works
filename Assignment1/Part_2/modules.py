import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {'weight': np.random.normal(loc=0, scale=0.1, size=(in_features, out_features)),
                       'bias': np.zeros(shape=out_features)
                       }
        self.grads = {'weight': np.zeros(shape=(in_features, out_features)),
                      'bias': np.zeros(shape=out_features)
                      }
        self.x = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        Implement the forward pass.
        """
        self.x = x
        res = np.dot(x, self.params['weight']) + self.params['bias']
        return res

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        Implement the backward pass.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0)
        d_loss = np.dot(dout, self.params['weight'].T)
        return d_loss


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        Implement the forward pass.
        """
        self.x = x
        res = np.maximum(0, x)
        return res

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        d_loss = dout
        d_loss[self.x < 0] = 0
        return d_loss


class SoftMax(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        Implement the forward pass using the Max Trick for numerical stability.
        """
        self.x = x
        x_max = np.max(x, axis=1, keepdims=True)
        y = np.exp(x - x_max)
        res = y / np.sum(y, axis=1, keepdims=True)
        return res

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout


class CrossEntropy(object):
    def __init__(self):
        self.DELTA = 1e-7

    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        Implement the forward pass.
        """
        res = -np.sum(y * np.log(x + self.DELTA)) / x.shape[0]
        return res

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        d_loss = (x + self.DELTA - y)
        return d_loss
