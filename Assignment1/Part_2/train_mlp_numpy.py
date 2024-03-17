import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here
    
    for step in range(max_steps):
        # TODO: Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights
        
        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            print(f"Step: {step}, Loss: ..., Accuracy: ...")
    
    print("Training complete!")

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
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
