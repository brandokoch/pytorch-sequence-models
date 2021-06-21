import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(GRUCell, self).__init__()

        self.relevance_gate = nn.Linear(n_input+n_hidden, n_hidden)
        self.update_gate = nn.Linear(n_input+n_hidden, n_hidden)
        self.linear = nn.Linear(n_hidden+n_hidden, n_hidden)

        self.out=nn.Linear(n_hidden,n_hidden)

    def forward(self, x, h):

        # Concatenate input and hidden state
        gates_in = torch.cat((h, x), dim=1)

        # Relevance gate decides which information from the hidden state is currently relevant
        relevance_gate_out = torch.sigmoid(self.relevance_gate(gates_in))

        # Update gate decides how much of the information from the previous hidden state will 
        # be overwritten
        update_gate_out = torch.sigmoid(self.update_gate(gates_in))

        # Hidden state candidates
        h_ = torch.cat((h * relevance_gate_out, x), dim=1)
        h_ = torch.tanh(self.linear(h_))

        # Perform hidden state updates
        h = (1-update_gate_out) * h + h_ * update_gate_out

        out=torch.relu(self.out(h))

        return out,h

class GRU(nn.Module):
    def __init__(self,n_input,n_hidden):
        super(GRU,self).__init__()

        self.n_input=n_input
        self.n_hidden=n_hidden

        self.GRUCell=GRUCell(n_input,n_hidden)

    def forward(self, input, h=None):

        # Input dims are (batch_size, seq_length, timestep_features)
        sequence_length=input.size()[1]

        # Initializing hidden state if not provided
        if h==None:
            h=torch.zeros((input.size()[0], self.n_hidden), device=input.device)

        outs=torch.tensor([],device=input.device)
        for i in range(sequence_length):
            x_timestep_features= torch.squeeze(input[:,i,:], dim=1)
            
            out, h = self.GRUCell(x_timestep_features, h)

            out=torch.unsqueeze(out,dim=1)
            outs=torch.cat((outs,out), dim=1)

        return outs
