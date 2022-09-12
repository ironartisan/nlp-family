#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2022/9/12 20:42 

import torch.nn as nn


class CBOW(nn.Module):
    """
    word2vec COBW
    """
    def __init__(self, vocab_size, embedding_dim, max_norm):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, _input):
        x= self.embeddings(_input)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram(nn.Module):
    """
    word2vec COBW
    """

    def __init__(self, vocab_size, embedding_dim, max_norm):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )
    def forward(self, _input):
        x = self.embeddings(_input)
        x = self.linear(x)
        return x