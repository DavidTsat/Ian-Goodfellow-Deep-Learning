#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, i, f, k=5, p=1):
        super(CNN, self).__init__()
        self.input_features = i
        self.output_features = f
        self.kernel_size = k
        self.padding = p
        self.conv1d = nn.Conv1d(in_channels=self.input_features, out_channels=self.output_features, kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x_reshaped):
        x_conv = self.conv1d(x_reshaped)
        return x_conv

# if __name__ == '__main__':
#     w_e_size = 10
#     e_char = 5
#     m_word = 12
#     model = CNN(e_char, w_e_size)
#     x_conv = torch.Tensor(16, e_char, m_word)
#     x_out = model(x_conv)
# #
#     xx = x_out