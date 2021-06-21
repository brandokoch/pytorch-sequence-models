import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self,n_input, n_hidden):
        super(RNNCell,self).__init__()

        self.linear1=nn.Linear(n_input+n_hidden, n_hidden)

        self.out=nn.Linear(n_hidden, n_hidden)

    def forward(self,x,h):
        h=torch.cat((h,x), dim=1)

        h=self.linear1(h)
        h=torch.relu(h)

        out=torch.relu(self.out(h))
        return out,h


class RNN(nn.Module):

    def __init__(self, n_input, n_hidden):
        super(RNN,self).__init__()
        
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.RNNCell=RNNCell(n_input, n_hidden)


    def forward(self, input, h=None):

        # Input dims are (batch_size, seq_length, timestep_features)
        sequence_length=input.size()[1]

        # Initializing hidden state if not provided
        if h==None:
            h=torch.zeros((input.size()[0], self.n_hidden), device=input.device)

        outs=torch.tensor([], device=input.device)
        for i in range(sequence_length):
            x_timestep_features=torch.squeeze(input[:,i,:], dim=1)
            out, h = self.RNNCell(x_timestep_features,h)

            out=torch.unsqueeze(out,dim=1)
            outs=torch.cat((outs,out), dim=1)

        return outs

        