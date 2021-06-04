import torch.nn as nn


class RNN_forward(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim   = hid_dim
        self.n_layers  = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn       = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.linear    = nn.Linear(hid_dim, input_dim)

    def forward(self, src):
        '''Input: input seq size = [batch * seq_len]'''

        embedded = self.dropout(self.embedding(src))
        '''Embed size = [batch * seq_len * embed_dim]'''

        '''Change embedded size to [seq_len * batch * embed_dim] (for RNN)
                     Output size = [seq_len * batch * hid_dim]'''
        outputs, (h_n, c_n) = self.rnn(embedded.permute(1, 0, 2))

        '''Apply linear layer to RNN outputs
        prediction size = [seq_len * batch * vocab_size ]'''
        prediction = self.linear(outputs.squeeze(0))

        return prediction, h_n, c_n