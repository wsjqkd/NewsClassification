import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    def __init__(self, num_words, num_classes, input_size, hidden_dim):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(num_words, input_size)
        self.net = None
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )


class LSTM(Model):
    def __init__(self, num_words, num_classes, input_size=64, hidden_dim=32, num_layer=1):
        super(LSTM, self).__init__(num_words, num_classes, input_size, hidden_dim)
        self.net = nn.LSTM(input_size, hidden_dim, num_layer, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embeding(x)
        pd = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.net(pd)
        pred = self.classification(cn[-1])
        return pred


class RNN(Model):
    def __init__(self, num_words, num_classes, input_size=64, hidden_dim=32, num_layer=1):
        super(RNN, self).__init__(num_words, num_classes, input_size, hidden_dim)
        self.net = nn.RNN(input_size, hidden_dim, num_layer, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embeding(x)
        pd = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        output, hn = self.net(pd)
        pred = self.classification(hn[-1])
        return pred

class GRU(Model):
    def __init__(self, num_words, num_classes, input_size=64, hidden_dim=32, num_layer=1):
        super(GRU, self).__init__(num_words, num_classes, input_size, hidden_dim)
        self.net = nn.GRU(input_size, hidden_dim, num_layer, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embeding(x)
        pd = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        output, hn = self.net(pd)
        pred = self.classification(hn[-1])
        return pred
