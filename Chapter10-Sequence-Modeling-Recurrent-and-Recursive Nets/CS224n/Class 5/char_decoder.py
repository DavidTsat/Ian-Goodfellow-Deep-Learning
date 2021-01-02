#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import pad_sents_char


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        char_embeds = self.decoderCharEmb(input)
        h_ts = []

        for i in range(char_embeds.shape[0]):
            x_t = char_embeds[i].unsqueeze(0)
            # if len(x_t.size())==2:
            #     x_t = x_t.unsqueeze(0)
            h_t, dec_hidden = self.charDecoder(x_t, dec_hidden)
            h_ts.append(h_t.squeeze())

        s_ts = [self.char_output_projection(h_t) for h_t in h_ts]

        return torch.stack(s_ts), dec_hidden
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        # pad_sents_char_padded = pad_sents_char(char_sequence, self.target_vocab.char_pad)

        loss = 0
        nll_loss = nn.CrossEntropyLoss()
        s_ts, dec_hidden = self.forward(char_sequence, dec_hidden)
        # char_sequence[:-1, :] = char_sequence[1:,:]
        # char_sequence[-1,:] = self.target_vocab.char_pad
        char_sequence = torch.roll(char_sequence, -1, 0)
        for i in range(s_ts.shape[0]):
            target = char_sequence[i]
            s_t = s_ts[i].clone()
            # s_t*(1 - (target == self.target_vocab.char_pad).int())
            for j, word in enumerate(target):
                if word == self.target_vocab.char_pad:
                    s_t[j,:] = torch.zeros_like(s_t[j,:].clone())
                    # s_t[j][self.target_vocab.char_pad] = torch.tensor(1, dtype=s_t.dtype)
                    # s_t[j, :] = 0
                    s_t[j][self.target_vocab.char_pad] = 1

            # print(i, s_ts.shape, 'AAAAAA', s_t.shape, target.shape)
            loss += nll_loss(s_t, target)
            # print(i, s_ts.shape, 'VVVVVV', s_t.shape, target.shape)
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        batch_size = initialStates[0].shape[1]
        current_char = torch.tensor([self.target_vocab.char2id['{'] for i in range(batch_size)], device=device).unsqueeze(0)
        output_word = [[] for i in range(batch_size)]
        for t in range(max_length):
            s_t_1, dec_hidden = self.forward(current_char, initialStates)
            # s_t_1 = self.char_output_projection(h_t_1)
            p_t_1 = torch.softmax(s_t_1, dim=-1)
            current_char = torch.argmax(p_t_1, dim=-1)

            # print('AAAAAAAAA', current_char,  len(current_char.size()))
            # if len(current_char.size())>1:
            for i in range(current_char.squeeze().shape[0]):
                if current_char.squeeze()[i] == self.target_vocab.char2id['{'] or current_char.squeeze()[i] == self.target_vocab.char2id['}']:
                    continue
                output_word[i].append(current_char.squeeze()[i])

        return output_word
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        ### END YOUR CODE

