#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = 50
        self.embedding = nn.Embedding(len(self.vocab.char2id), self.e_char).to("cuda")
        self.cnn = CNN(self.e_char, word_embed_size).to("cuda")
        self.relu = nn.ReLU()
        self.highway = Highway(self.word_embed_size).to("cuda")
        self.dropout = nn.Dropout(0.3)
        ### YOUR CODE HERE for part 1h

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        input = input.to("cuda")
        x_embeds = self.embedding(input.permute(1,0,2))
        x_reshapeds = x_embeds.permute(0,1,3,2).contiguous()

        x_cnn_based_embeds = torch.zeros(input.shape[0], input.shape[1], self.word_embed_size).to("cuda")

        for i in range(x_reshapeds.shape[1]):
            batch_words = x_reshapeds[:, i, :, :]
            batch_words_convs = self.relu(self.cnn(batch_words))
            batch_words_convs, _ = torch.max(batch_words_convs, dim=-1) # conv_out: (batch_size, e_word)
            x_cnn_based_embeds[i] = self.dropout(self.highway(batch_words_convs))

        return x_cnn_based_embeds
        ### END YOUR CODE