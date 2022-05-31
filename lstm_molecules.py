import torch
import numpy as np
import random
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
import time
from torch.nn.functional import softmax
import wandb

'''Custom modules'''
from data_ import Dataset_
from model_ import RNN_forward
from utils_ import loss_,test_loss,sample_,model_quality,sweep_to_dict


dir_config = '/content/LSTM/param_dict.yaml'
dir_dataset = '/content/LSTM/data/qm9_smiles.txt'

wandb.init(config=sweep_to_dict(dir_config))
config = wandb.config


'''Set the random seeds for deterministic results'''
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

'''Set up train and test dataloaders'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Dataset_(dir_dataset,device)
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
print('Train size ={}, Test size ={}'.format(train_size,test_size))
train_, test_ = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(SEED))
train_l, test_l = DataLoader(train_,batch_size=config.batch_size,drop_last=True),DataLoader(test_,batch_size=config.batch_size,drop_last=True)
print('Train batches ={}'.format(len(train_l)))
model = RNN_forward(input_dim=len(dataset.dict_)+1,emb_dim=config.EMBED_DIM,hid_dim=config.n_hidden, \
                    n_layers=config.n_layers,layer_norm=False,dropout=config.drop1)
model.to(device)
opt = torch.optim.Adam(model.parameters(),lr=config.LR)

hidden = model.init_hidden_(config.batch_size,device)


def sample_(model,n_molecules,temperature,dataset,device,seq_len=100):
    '''Input:
        1) y_t unnormilised logits for the next symbols
        2) temperature - sampling temperature
        Return:
            next token
        https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/sampling.html'''
    model.eval()
    hidden = model.init_hidden_(n_molecules, device)
    x      = torch.tensor([dataset.dict_['G']]*n_molecules, dtype=torch.long, device=device).unsqueeze(0).permute(1,0)
    seq    = torch.ones((n_molecules,1), device=device)

    for i in range(seq_len):
        logits, hidden = model(x,hidden)
        prob           = softmax(logits / temperature, dim=-1)
        x              = torch.multinomial(prob, 1)
        seq            = torch.hstack([seq,x])

    matrix      =  np.vectorize(dataset.dict_inv.get)(seq.detach().cpu().numpy()[:,1:])
    smiles_list = []
    for i in range(n_molecules):
        string_ = ''.join(list(matrix[i, :]))
        smiles_list.append(string_.split('E')[0])

    return smiles_list

counter = 0

for j in range(config.epochs):
    for i,(input_seq,target_seq,mask) in enumerate(train_l):
        model.train()
        opt.zero_grad()
        pred,_= model(input_seq,hidden)
        l        = loss_(pred,target_seq,mask)
        l.backward()
        opt.step()

        '''Log train\test losses every 1/2 epoch'''
        if counter%(int(len(train_l)/2)) == 0:

            wandb.log({'Train_Loss':l,'epoch':round(counter/len(train_l),2)})
            wandb.log({'Test_loss':test_loss(test_l,model,config.batch_size,device),'epoch':round(counter/len(train_l),2)})

        counter+=1