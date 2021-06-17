import torch
from data_ import Dataset_
from torch.nn.functional import softmax
import numpy as np
import json
from rdkit import Chem
from utils_ import sample_

rnn = torch.load(r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\model1.pth',map_location=torch.device('cpu') )
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_dataset     = r'C:\Users\zcemg08\PycharmProjects\LSTM\data\qm9_smiles.txt'
dataset         = Dataset_(dir_dataset,device)

with open(r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict1.json', 'r') as fp:
    dict1 = json.load(fp)

with open(r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict_inv1.json', 'r') as fp:
    dict_inv1 = json.load(fp)


dict_inv1 = {int(key):value for key,value in dict_inv1.items()}
dict1     = {key:int(value) for key,value in dict1.items()}

n_steps = 1000
m_sequences = 500

dataset.dict_inv = dict_inv1
dataset.dict_ = dict1
smiles_bank = []

func = substructure

for i in range(10):

    Y = sample_(model, m_sequences, 1, dict1, dict_inv1, device)
    smiles_bank += Y
    top_smiles = top_k_smiles(smiles_bank, func, k=64)
    smiles_bank = top_smiles

    target_seq_ = torch.zeros((1, dataset.max_l + 1), device=device)
    input_seq = torch.zeros((1, dataset.max_l + 1), device=device)
    mask = torch.zeros((1, dataset.max_l + 1), device=device)

    for s in top_smiles:
        seq = dataset.seq_(s)
        t_ = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
        i_ = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        m_ = dataset.msk_(len(s)).unsqueeze(0)

        target_seq_ = torch.vstack([target_seq_, t_])
        input_seq = torch.vstack([input_seq, i_])
        mask = torch.vstack([mask, m_])

    target_seq_ = target_seq_[1:, :]
    input_seq = input_seq[1:, :]
    mask = mask[1:, :]



