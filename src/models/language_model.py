import torch
import torch.nn as nn
import sys
from models.rnn import RNN
from models.lstm import LSTM
from models.gru import GRU

class LanguageModel(nn.Module):
    def __init__(self,vocab_sz,n_hidden, rnn_type='RNN'):
        super(LanguageModel,self).__init__()

        self.embedding=nn.Embedding(vocab_sz,n_hidden)

        assert rnn_type in ['RNN','LSTM','GRU'], "Unsupported rnn_type"
        self.rnn=getattr(sys.modules[__name__], rnn_type)(n_hidden,n_hidden)

        self.dropout=nn.Dropout(0.2)

        self.linear=nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        x=self.embedding(x)
        x =self.rnn(x)

        x=self.dropout(x)

        out=self.linear(x)

        # We dont apply softmax to output since nn.CrossEntropyLoss 
        # combines LogSoftmax and NLLLoss
        return out.permute(0,2,1)