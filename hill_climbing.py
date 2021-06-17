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


def substructure(m,fragment):
    mol = Chem.MolFromSmiles(m)
    if mol is None:
        return 0
    else:
        if len(mol.GetSubstructMatches(Chem.MolFromSmiles(fragment)))>0:
            return 1
        else:
            return 0

frag = 'CC(C)=O'

Y = sample_(rnn,1000,1,dict1,dict_inv1,device)

print(sum([substructure(x,frag) for x in Y]))



