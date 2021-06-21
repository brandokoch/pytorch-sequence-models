import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(LSTMCell, self).__init__()

        # WARNING: this implementation is the most intuitive but 
        # not the most efficient. 4 Linear layer multiplications can be 
        # condensed into one which would speed up training.
        self.forget_gate = nn.Linear(n_input+n_hidden, n_hidden)
        self.input_gate = nn.Linear(n_input+n_hidden, n_hidden)
        self.cell_gate = nn.Linear(n_input+n_hidden, n_hidden)
        self.output_gate = nn.Linear(n_input+n_hidden, n_hidden)

        self.out=nn.Linear(n_hidden, n_hidden)

    def forward(self, x, state):

        h, c = state

        # Concatenate input and hidden state
        h = torch.cat([h, x], dim=1)

        # Forget gate decides what to eliminate from the cell state.
        forget_gate_out = torch.sigmoid(self.forget_gate(h))

        # Input gate decides what needs to be updated and to what degree.
        input_gate_out = torch.sigmoid(self.input_gate(h))

        # Cell gate decides what are the updated values for the positions chosen by the input gate
        cell_gate_out = torch.sigmoid(self.cell_gate(h))

        # Output gate decides which information from the cell state is relevant for the next hidden state
        output_gate_out = torch.sigmoid(self.output_gate(h))

        c = c*forget_gate_out
        c = c+(input_gate_out*cell_gate_out)

        h = output_gate_out*torch.tanh(c)
        
        out= torch.relu(self.out(h))

        return out, (h, c)


class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(LSTM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.LSTMCell = LSTMCell(n_input, n_hidden)

    def forward(self, input, state=None):

        # Input dims are (batch_size, seq_length, timestep_features)
        sequence_length = input.size()[1]

        # Initialize hidden and cell state if not provided
        if state == None:
            h, c = (torch.zeros((input.size()[0], self.n_hidden), device=input.device),
                    torch.zeros((input.size()[0], self.n_hidden), device=input.device))
        else:
            h, c = state

        outs = torch.tensor([], device=input.device)
        for i in range(sequence_length):
            x_timestep_features = torch.squeeze(input[:, i, :], dim=1)

            out, (h, c) = self.LSTMCell(x_timestep_features, (h, c))

            out = torch.unsqueeze(out, dim=1)
            outs = torch.cat((outs, out), dim=1)

        return outs
