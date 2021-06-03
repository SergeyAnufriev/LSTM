import torch
import numpy as np
import random
from torch.utils.data import DataLoader,random_split
'''Custom modules'''
from data_ import Dataset_
from model_ import RNN_forward
from utils_ import loss_

'''Model parameters'''
EMBED_DIM  = 100
batch_size = 64
n_hidden   = 512
n_layers   = 2
drop1      = 0.3
drop2      = 0.5
LR         = 10e-3

'''Set the random seeds for deterministic results'''
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

dir_dataset     = r'C:\Users\zcemg08\PycharmProjects\LSTM\data\qm9_smiles.txt'
dataset         = Dataset_(dir_dataset)
train_size      = int(len(dataset)*0.8)
test_size       = len(dataset) - train_size
print('Train size ={}, Test size ={}'.format(train_size,test_size))

train_, test_   = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(SEED))
train_l, test_l = DataLoader(train_,batch_size=batch_size),DataLoader(test_,batch_size=batch_size)


model = RNN_forward(input_dim=len(dataset.dict_)+1,emb_dim=EMBED_DIM,hid_dim=n_hidden,n_layers=n_layers,dropout=drop1)
opt   = torch.optim.Adam(model.parameters(),lr=LR)


for i,(input_seq,target_seq,mask) in enumerate(train_l):

    opt.zero_grad()
    pred = model(input_seq)
    l    = torch.mean(torch.sum(loss_(pred,target_seq,mask), dim=-1))
    l.backward()
    opt.step()


    print('Train loss = {}'.format(l))

