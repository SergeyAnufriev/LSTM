import torch
import torch.nn as nn


class RNN_forward(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout,layer_norm=True):
        super().__init__()

        self.hid_dim   = hid_dim
        self.n_layers  = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn       = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.linear    = nn.Linear(hid_dim, input_dim)

        self.p1, self.p2  = self.network(input_dim,emb_dim, hid_dim,layer_norm)

    def network(self, input_dim,emb_dim, hid_dim,layer_norm):
        '''creates lstm network with/without layer normalisation'''

        part1 = nn.ModuleList()
        part2 = nn.ModuleList()

        part1.append(nn.Embedding(input_dim, emb_dim))
        part2.append(nn.Linear(hid_dim, input_dim))

        if layer_norm == True:
            part1.append(nn.LayerNorm(emb_dim))
            part2.insert(0,nn.LayerNorm(hid_dim))

        return nn.Sequential(*part1),nn.Sequential(*part2)

    def forward(self,scr,hidden):
        '''Input: input
        1) sequence with size = [batch * seq_len]
        2) hidden state (h_0,c_0)'''

        embedded             = self.dropout(self.p1(scr))
        '''           embed size = [batch * seq_len * embed_dim]'''

        '''Change embedded size to [seq_len * batch * embed_dim] (for RNN)
                     Output size = [seq_len * batch * hid_dim]'''
        outputs, (h_n, c_n)  = self.rnn(embedded.permute(1, 0, 2), hidden)

        '''Apply linear layer to RNN outputs
                prediction size = [seq_len * batch * vocab_size ]'''
        prediction           = self.p2(outputs.squeeze(0))

        return prediction, (h_n, c_n)


    def forward_100(self,src,hidden):
        '''Input: input
        1) sequence with size = [batch * seq_len]
        2) hidden state (h_0,c_0)'''

        embedded = self.dropout(self.embedding(src))
        '''Embed size = [batch * seq_len * embed_dim]'''

        '''Change embedded size to [seq_len * batch * embed_dim] (for RNN)
                     Output size = [seq_len * batch * hid_dim]'''
        outputs, (h_n, c_n) = self.rnn(embedded.permute(1, 0, 2),hidden)

        '''Apply linear layer to RNN outputs
        prediction size = [seq_len * batch * vocab_size ]'''
        prediction = self.linear(outputs.squeeze(0))

        return prediction, (h_n, c_n)

    def init_hidden_(self,batch_size,device):
        '''Initialise hidden and cell states'''

        h_0 = torch.zeros((self.n_layers,batch_size,self.hid_dim),device=device)
        c_0 = torch.zeros((self.n_layers,batch_size,self.hid_dim),device=device)

        return h_0,c_0