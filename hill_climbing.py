import torch
from data_ import Dataset_
from torch.nn.functional import softmax
import numpy as np
import json
from rdkit import Chem
from utils_ import sample_,loss_,model_quality
from chem_ import top_k_smiles
import os
from chem_ import log_p
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

dir_dataset     = r'C:\Users\zcemg08\PycharmProjects\LSTM\data\qm9_smiles.txt'
dir_model       = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\model1.pth'
dir_dict        = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict1.json'
dir_dict_inv    = r'C:\Users\zcemg08\PycharmProjects\LSTM\files_\dict_inv1.json'

n_steps = 1000
m_sequences = 200
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

'''Database molecules'''
f         = open(r'C:\Users\zcemg08\PycharmProjects\LSTM\data\/qm9_smiles.txt','r')
data_mols = [line.split(',')[0][:-1] for line in f.readlines()]

dataset.dict_inv = dict_inv1
dataset.dict_ = dict1
smiles_bank = []

model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
progress_ = []

hidden = model.init_hidden_(64, device)

print('maximum length of generated smiles ={}'.format(dataset.max_l))

Y = sample_(model, m_sequences, 1, dict1, dict_inv1, device)
valid,unique,novel = model_quality(Y,data_mols)

with open(r'C:\Users\zcemg08\PycharmProjects\LSTM\results\Output.txt', "w") as text_file:
    text_file.write('Results at iter = {}: valid = {},unique = {}, novel = {}'.format(0, valid, unique, novel))

'''Plot Distr Of molecules before fine tuning'''
log_p_vals_before_fine_tune = [log_p(x) for x in Y]
while -10**3 in log_p_vals_before_fine_tune:
    log_p_vals_before_fine_tune.remove(-10**3)

plt.hist(log_p_vals_before_fine_tune)
plt.savefig(r'C:\Users\zcemg08\PycharmProjects\LSTM\results\hist_0.png')


for i in range(110):

    Y = sample_(model, m_sequences, 1, dict1, dict_inv1, device)
    smiles_bank += Y
    top_smiles = top_k_smiles(smiles_bank, log_p, k=64)
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

    if i % 10 == 0 and i!=0:
        log_p_vals_ = [log_p(x) for x in Y]
        while -10 ** 3 in log_p_vals_:
            log_p_vals_.remove(-10 ** 3)
        plt.hist(log_p_vals_)
        plt.savefig(r'C:\Users\zcemg08\PycharmProjects\LSTM\results\hist_%d.png'%i)

        '''Save progress per fine tune step'''
        valid, unique, novel = model_quality(Y, data_mols)
        with open(r'C:\Users\zcemg08\PycharmProjects\LSTM\results\Output.txt', "a") as text_file:
            text_file.write('\n')
            text_file.write('Results at iter = {}: valid = {},unique = {}, novel = {}'.format(i, valid, unique, novel))









