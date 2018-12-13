
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    """LSTM class"""
    def __init__(self, input_size, hidden_size, output_size):
        '''

        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        '''
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        #LSTM
        self.lstm = nn.LSTM(input_size, hidden_size).to(device)
        self.hidden2Cat = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, input):

        lstm_out, self.hidden = self.lstm(input, self.hidden)
        output = self.hidden2Cat(lstm_out[-1]) #many to one
        output = F.log_softmax(output, dim=1)

        return output

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))
