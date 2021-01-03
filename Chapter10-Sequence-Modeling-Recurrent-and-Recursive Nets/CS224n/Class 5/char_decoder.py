#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size).to("cuda")
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id)).to("cuda")
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad).to("cuda")

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        input = input.to("cuda")
        char_embeds = self.decoderCharEmb(input)
        h_ts = []

        if dec_hidden and not dec_hidden[0].is_cuda:
            dec_hidden = (dec_hidden[0].to("cuda"), dec_hidden[1].to("cuda"))

        for i in range(char_embeds.shape[0]):
            x_t = char_embeds[i].unsqueeze(0)
            h_t, dec_hidden = self.charDecoder(x_t, dec_hidden)
            h_ts.append(h_t[0])

        s_ts = [self.char_output_projection(h_t) for h_t in h_ts]

        return torch.stack(s_ts).to("cuda"), dec_hidden
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
        char_sequence = char_sequence.to("cuda")
        loss = 0
        nll_loss = nn.CrossEntropyLoss(reduction='mean').to("cuda")
        target_sequence = char_sequence[1:]
        char_sequence[char_sequence == self.target_vocab.end_of_word] = self.target_vocab.char_pad
        x_sequence = char_sequence[:-1]
        s_ts, dec_hidden = self.forward(x_sequence, dec_hidden)
        for i in range(s_ts.shape[0]):
            target = target_sequence[i]
            # masking worsens training, maybe trying without mask
            loss = loss + nll_loss(s_ts[i], target)

            # pad_char_mask = (target != self.target_vocab.char_pad).float().unsqueeze(1)
            # predicted = s_ts[i].clone() * pad_char_mask
            # predicted[~pad_char_mask.byte().squeeze(), self.target_vocab.char_pad] = 1
            # loss = loss + nll_loss(predicted, target)

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
        current_char = torch.tensor([self.target_vocab.start_of_word for i in range(batch_size)], device=device).unsqueeze(0)
        output_word = ['' for i in range(batch_size)]
        dec_hidden = initialStates
        for t in range(max_length):
            s_t_1, dec_hidden = self.forward(current_char, dec_hidden)
            p_t_1 = torch.softmax(s_t_1, dim=-1)
            current_char = torch.argmax(p_t_1, dim=-1)

            for i in range(batch_size):
                if current_char[0][i] == self.target_vocab.start_of_word or current_char[0][i] == self.target_vocab.end_of_word or \
                        current_char[0][i] == self.target_vocab.char_pad:
                    continue
                else:
                    output_word[i] += self.target_vocab.id2char[current_char[0][i].item()]
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

