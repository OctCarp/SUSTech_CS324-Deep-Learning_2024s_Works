from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch.utils.data as data


class PalindromeDataset(data.Dataset):

    def __init__(self, input_length, total_len, one_hot=False):
        """
        Args:
            seq_length: length of the sequence(both input and target)
            total_len: total number of samples in the dataset
            one_hot: whether to use one-hot encoding or not
        """
        self.input_length = input_length
        self.seq_length = input_length+1
        self.one_hot = one_hot
        self.half_length = math.ceil(self.seq_length/2)
        max_num = 10 ** self.half_length
        self.total_len = total_len
        if self.total_len > max_num:
            print("Warning: total_len is larger than the maximum possible length. ")
            print("Setting total_len to the maximum possible length. ")
            print(
                "Warning: access length of dataset by len(dataset) to get the actual length. ")
            self.total_len = 10 ** self.half_length
        self.data = np.random.default_rng().choice(
            max_num, self.total_len, replace=False)
        self.mapping = np.eye(10) if one_hot else np.arange(10).reshape(10, 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Keep last digit as target label.
        full_palindrome = self.generate_palindrome(self.data[idx])
        # Split palindrome into inputs (N-1 digits) and target (1 digit)
        inputs, labels = full_palindrome[0:-1], int(full_palindrome[-1])
        inputs = self.mapping[inputs].astype(np.float32)
        return inputs, labels

    def generate_palindrome(self, data):
        data = tuple(map(int, str(data)))
        left = np.zeros(self.half_length).astype(np.uint64)
        left[-len(data):] = data
        if self.seq_length % 2 == 0:
            right = left[::-1]
        else:
            right = left[-2::-1]
        return np.concatenate((left, right))
