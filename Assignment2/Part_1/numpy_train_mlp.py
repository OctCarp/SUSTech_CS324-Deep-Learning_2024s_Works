import argparse
import numpy as np

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 800


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


class NumpyMLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes the multi-layer perceptron object.

        This function should initialize the layers of the MLP including any linear layers and activation functions
        you plan to use. You will need to create a list of linear layers based on n_inputs, n_hidden, and n_classes.
        Also, initialize ReLU activation layers for each hidden layer and a softmax layer for the output.

        Args:
            n_inputs (int): Number of inputs (i.e., dimension of an input vector).
            n_hidden (list of int): List of integers, where each integer is the number of units in each hidden layer.
            n_classes (int): Number of classes of the classification problem (i.e., output dimension of the network).
        """
        # Hint: You can use a loop to create the necessary number of layers and add them to a list.
        # Remember to initialize the weights and biases in each layer.
        self.layers = []
        prev_size = n_inputs
        for unit_size in n_hidden:
            self.layers.append(Linear(prev_size, unit_size))
            self.layers.append(ReLU())
            prev_size = unit_size
        self.layers.append(Linear(prev_size, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x):
        """
        Predicts the network output from the input by passing it through several layers.

        Here, you should implement the forward pass through all layers of the MLP. This involves
        iterating over your list of layers and passing the input through each one sequentially.
        Don't forget to apply the activation function after each linear layer except for the output layer.

        Args:
            x (numpy.ndarray): Input to the network.

        Returns:
            numpy.ndarray: Output of the network.
        """
        # Start with the input as the initial output
        out = x

        # Implement the forward pass through each layer.
        # Hint: For each layer in your network, you will need to update 'out' to be the layer's output.
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        """
        Performs the backward propagation pass given the loss gradients.

        Here, you should implement the backward pass through all layers of the MLP. This involves
        iterating over your list of layers in reverse and passing the gradient through each one sequentially.
        You will update the gradients for each layer.

        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        # Implement the backward pass through each layer.
        # Hint: You will need to update 'dout' to be the gradient of the loss with respect to the input of each layer.

        # No need to return anything since the gradients are stored in the layers.

        for layer in reversed(self.layers):
            dout = layer.backward(dout)


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    targets_classes = np.argmax(targets, axis=1)
    correct = np.sum(predicted_classes == targets_classes)
    acc = correct / predictions.shape[0]
    return acc


def train(data, batch_size=BATCH_SIZE_DEFAULT, dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT,
          learning_rate=LEARNING_RATE_DEFAULT, max_steps=MAX_EPOCHS_DEFAULT, eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        data: training data
        batch_size: Int for batch size
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set

    NOTE: Add necessary arguments such as the data, your model...
    """
    np.random.seed(42)

    # Load your data here
    train_x, train_y, test_x, test_y = data

    # Initialize your MLP model and loss function (CrossEntropy) here
    dnn_hidden_units = list(map(int, dnn_hidden_units.split(',')))
    mlp = NumpyMLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)
    loss_fn = CrossEntropy()

    train_length = train_x.shape[0]
    eval_step_l = []
    train_acc_l = []
    test_acc_l = []
    train_loss_l = []
    test_loss_l = []
    batch_sz = np.minimum(batch_size, train_length)
    batch_sz = np.maximum(batch_sz, -train_length)

    for step in range(max_steps):
        train_acc, train_loss = 0, 0
        random_index = np.random.permutation(train_length)
        shuffled_x, shuffled_y = train_x[random_index], train_y[random_index]
        if batch_sz < 0:
            actual_batch = -batch_sz
            index = np.random.randint(low=0, high=train_length - actual_batch + 1)
            step_x = shuffled_x[index:index + actual_batch]
            step_y = shuffled_y[index:index + actual_batch]
            predictions = mlp.forward(step_x)
            dout = loss_fn.backward(predictions, step_y)
            mlp.backward(dout)
            train_acc = accuracy(predictions, step_y)
            # train_loss = loss_fn.forward(predictions, step_y)
            for layer in mlp.layers:
                if isinstance(layer, Linear):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']
        else:

            for start_idx in range(0, train_length - batch_sz + 1, batch_sz):
                step_x = shuffled_x[start_idx:start_idx + batch_sz]
                step_y = shuffled_y[start_idx:start_idx + batch_sz]
                predictions = mlp.forward(step_x)
                dout = loss_fn.backward(predictions, step_y)
                mlp.backward(dout)
                train_acc += accuracy(predictions, step_y)
                train_loss += loss_fn.forward(predictions, step_y)

                for layer in mlp.layers:
                    if isinstance(layer, Linear):
                        layer.params['weight'] -= learning_rate * layer.grads['weight'] / batch_sz
                        layer.params['bias'] -= learning_rate * layer.grads['bias'] / batch_sz
            iter_cnt = train_length // batch_sz
            train_acc /= iter_cnt
            train_loss /= iter_cnt

        if step % eval_freq == 0 or step == max_steps - 1:
            test_predictions = mlp.forward(test_x)
            test_acc = accuracy(test_predictions, test_y)
            test_loss = loss_fn.forward(test_predictions, test_y)
            # # print(f"Step: {step}, Loss: {test_loss:.6f}, Accuracy: {test_acc:.2f}%")

            eval_step_l.append(step)
            train_acc_l.append(train_acc)
            test_acc_l.append(test_acc)
            train_loss_l.append(train_loss)
            test_loss_l.append(test_loss)

    return eval_step_l, train_acc_l, test_acc_l, train_loss_l, test_loss_l
