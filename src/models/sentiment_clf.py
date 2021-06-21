import torch
import torch.nn as nn
import sys
from models.rnn import RNN
from models.lstm import LSTM
from models.gru import GRU


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_sz, n_hidden, rnn_type='RNN'):
        super(SentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_sz, n_hidden)

        assert rnn_type in ['RNN', 'LSTM', 'GRU'], "Unsupported rnn_type"
        self.rnn = getattr(sys.modules[__name__], rnn_type)(n_hidden, n_hidden)

        self.dropout = nn.Dropout(0.2)

        self.linear = nn.Linear(n_hidden*2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        x = self.dropout(x)

        # Using the avg and max pool of all RNN outputs
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, 1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        out = torch.cat((avg_pool, max_pool), dim=1)
        out = self.linear(out)

        # We dont apply sigmoid to output since nn.BCEWithLogitsLoss
        # combines a Sigmoid layer and the BCELoss
        return torch.squeeze(out, dim=1)


class SentimentClassifier1(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        super(SentimentClassifier1, self).__init__()

        self.embedding = nn.Embedding(vocab_sz, n_hidden)

        self.rnn = nn.LSTM(n_hidden, n_hidden, batch_first=True)

        self.linear = nn.Linear(n_hidden*2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        # Using the avg and max pool of all RNN outputs
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, 1)

        # We concatenate them (hidden size before the linear layer is multiplied by 2)
        out = torch.cat((avg_pool, max_pool), dim=1)
        out = self.linear(out)

        # We dont apply sigmoid to output since nn.BCEWithLogitsLoss
        # combines a Sigmoid layer and the BCELoss
        return torch.squeeze(out, dim=1)
