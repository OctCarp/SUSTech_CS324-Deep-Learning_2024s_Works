from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes, *args, **kwargs):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super().__init__(*args, **kwargs)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        hidden_input = n_inputs
        for hidden_size in n_hidden:
            self.layers.append(nn.Linear(hidden_input, hidden_size))
            self.layers.append(nn.ReLU())
            hidden_input = hidden_size
        self.layers.append(nn.Linear(n_hidden[-1], n_classes))
        self.layers.append(nn.LogSoftmax(dim=1))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        x = x.view(-1, self.n_inputs)
        out = self.network(x)
        return out
