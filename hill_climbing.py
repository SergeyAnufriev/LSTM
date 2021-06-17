import torch
from data_ import Dataset_
from torch.nn.functional import softmax
import numpy as np
import json
from rdkit import Chem
from utils_ import sample_,loss_
from chem_ import top_k_smiles,substructure


dir_dataset     = r'C:\Users\zcemg08\PycharmProjects\LSTM\data\qm9_smiles.txt'
dir_model       = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\model1.pth'
dir_dict        = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict1.json'
dir_dict_inv    = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict_inv1.json'

n_steps = 1000
m_sequences = 500
LR = 10e-3

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = torch.load(dir_model,map_location=device)
dataset         = Dataset_(dir_dataset,device)

with open(dir_dict, 'r') as fp:
    dict1 = json.load(fp)

with open(dir_dict_inv, 'r') as fp:
    dict_inv1 = json.load(fp)

dict_inv1 = {int(key):value for key,value in dict_inv1.items()}
dict1     = {key:int(value) for key,value in dict1.items()}

dataset.dict_inv = dict_inv1
dataset.dict_ = dict1
smiles_bank = []

model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
progress_ = []

hidden = model.init_hidden_(64, device)

def ration(model,func):
  model.eval()
  n_substruct = sum([func(x) for x in sample_(model,1000,1,dict1,dict_inv1,device)])
  return n_substruct/1000

for i in range(1000):

    Y = sample_(model, m_sequences, 1, dict1, dict_inv1, device)
    smiles_bank += Y
    top_smiles = top_k_smiles(smiles_bank, substructure, k=64)
    smiles_bank = top_smiles

    target_seq_ = torch.zeros((1, dataset.max_l + 1), device=device, dtype=torch.long)
    input_seq = torch.zeros((1, dataset.max_l + 1), device=device, dtype=torch.long)
    mask = torch.zeros((1, dataset.max_l + 1), device=device, dtype=torch.long)

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

    model.train()
    opt.zero_grad()
    pred, _ = model(input_seq, hidden)
    l = loss_(pred, target_seq_, mask)
    l.backward()
    opt.step()

    if i % 100 == 0:
        print('Loss = {}'.format(l))
        r = ration(model, substructure)
        progress_.append(r)
        print('ratio ketone group before train ={}'.format(r))




