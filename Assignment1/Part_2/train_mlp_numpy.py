import argparse
import numpy as np
from mlp_numpy import MLP
from modules import Linear, CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 800
IS_SGD_DEFAULT = False


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
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
    mlp = MLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)
    loss_fn = CrossEntropy()

    train_length = train_x.shape[0]
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    batch_sz = np.minimum(batch_size, train_length)
    batch_sz = np.maximum(batch_sz, -train_length)

    for step in range(max_steps):
        # Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights
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
            train_loss = loss_fn.forward(predictions, step_y)
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
            # Evaluate the model on the test set
            # 1. Forward pass on the test set
            test_predictions = mlp.forward(test_x)
            # 2. Compute loss and accuracy
            test_acc = accuracy(test_predictions, test_y) * 100
            test_loss = loss_fn.forward(test_predictions, test_y)
            # print(f"Step: {step}, Loss: {test_loss:.6f}, Accuracy: {test_acc:.2f}%")

            train_acc_list.append([step, train_acc * 100])
            train_loss_list.append([step, train_loss])
            test_acc_list.append([step, test_acc])
            test_loss_list.append([step, test_loss])

    print(f'Batch {batch_sz} Training complete!')
    return train_acc_list, train_loss_list, test_acc_list, test_loss_list


def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Number of batch size for train')
    FLAGS = parser.parse_known_args()[0]
    data = None
    train(data, FLAGS.batch_size, FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)


if __name__ == '__main__':
    main()
