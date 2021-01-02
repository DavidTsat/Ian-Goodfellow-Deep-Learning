#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()

        # self.W_proj = nn.Parameter(torch.Tensor((word_embed_size, word_embed_size)))
        # self.b_proj = nn.Parameter(torch.Tensor((1, word_embed_size)))
        #
        # self.W_gate = nn.Parameter(torch.Tensor((word_embed_size, word_embed_size)))
        # self.b_gate = nn.Parameter(torch.Tensor((1, word_embed_size)))
        self.proj_linear = nn.Linear(word_embed_size, word_embed_size, bias=True).to("cuda")
        self.gate_linear = nn.Linear(word_embed_size, word_embed_size, bias=True).to("cuda")
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x_conv):
        x_proj = self.relu(self.proj_linear(x_conv))
        x_gate = self.sigm(self.gate_linear(x_conv))

        x_highway = torch.mul(x_gate, x_proj) + torch.mul((torch.ones_like(x_gate) - x_gate), x_conv)

        return x_highway


# if __name__ == '__main__':
#     w_e_size = 10
#     model = Highway(w_e_size)
#     x_conv = torch.Tensor(16, w_e_size)
#     x_out = model(x_conv)
#
#     xx = x_out