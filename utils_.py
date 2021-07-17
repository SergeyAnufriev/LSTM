import torch
from torch.nn.functional import softmax
import numpy as np
from rdkit import Chem

losses = torch.nn.CrossEntropyLoss(reduction='none')

def loss_(logits,target,mask):

    ''' Input - 1,2,3
        1) logits unnormilised model predictions       : size = seq_l * bz  * n_classes
        2) target true label for each item in sequence : size = bz * seq_l
        3) mask - 1 for any symbol except for PAD is 0 : size = bz * seq_l
        Output loss for each symbol                    : size = bz * seq_l'''

    '''Change logits size to bz * n_classes * seq_l
    https://stackoverflow.com/questions/63648735/pytorch-crossentropy-loss-with-3d-input'''
    logits = logits.permute(1,2,0)
    l      = losses(logits,target)*mask
    '''Devide loss for each sequence by the number of unpadded symbols
    https://discuss.pytorch.org/t/calculating-loss-on-sequences-with-variable-lengths/9891'''
    l      = torch.sum(l, dim=-1).unsqueeze(-1)/torch.sum(mask,dim=-1).unsqueeze(-1)

    '''Return average loss for molecules in batch'''
    return torch.mean(l)



def test_loss(test_l,model,batch_size,device):
    '''Input:
    1) test_l - test data loader
    2) model  - model to be tested]
    Output: average loss on the whole dataset'''

    model.eval()
    losses = 0
    hidden = model.init_hidden_(batch_size, device)

    for input_seq, target_seq, mask in test_l:

        pred,_     = model(input_seq,hidden)
        l          = loss_(pred,target_seq,mask)
        losses += l

    return losses/len(test_l)


def softmax_temp(y_t,temperature):
    '''Input:
    1) y_t unnormilised logits for the next symbols
    2) temperature - sampling temperature
    Return:
        next token
    https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/sampling.html'''

    prob = softmax(y_t/temperature,dim=-1)
    return torch.multinomial(prob,1)[0]

def sample_(model,n_molecules,temperature,dict_,dict_inv,device,seq_len=100):
    '''Input:
        1) y_t unnormilised logits for the next symbols
        2) temperature - sampling temperature
        Return:
            next token
        https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/sampling.html'''
    model.eval()
    hidden = model.init_hidden_(n_molecules, device)
    x      = torch.tensor([dict_['G']]*n_molecules, dtype=torch.long, device=device).unsqueeze(0).permute(1,0)
    seq    = torch.ones((n_molecules,1), device=device)

    for i in range(seq_len):
        logits, hidden = model(x,hidden)
        prob           = softmax(logits / temperature, dim=-1)
        x              = torch.multinomial(prob, 1)
        seq            = torch.hstack([seq,x])

    matrix      =  np.vectorize(dict_inv.get)(seq.detach().cpu().numpy()[:,1:])
    smiles_list = []
    for i in range(n_molecules):
        string_ = ''.join(list(matrix[i, :]))
        smiles_list.append(string_.split('E')[0])

    return smiles_list

def valid_(smi):

  '''Input: smiles string
  output binary: 1-valid, 0-not valid molecule'''

  m = Chem.MolFromSmiles(smi,sanitize=False)
  if m is None:
    return 0
  else:
    try:
      mol = Chem.SanitizeMol(m)
      return 1
    except:
      return 0

def model_quality(sampled_molecules,data_molecules):

    '''Input:
    1) sampled_molecules: list of smiles strings generated
    2) data_molecules: list of smiles strings in dataset

    Return:
        valid molecules %, unique molecules % and novel molecules %'''

    valid_mols      = [x for x in sampled_molecules if valid_(x)==1]
    unique_mols     = set(valid_mols)
    novel_mols      = unique_mols-set(data_molecules)
    S               = len(sampled_molecules)

    return len(valid_mols)/S,len(unique_mols)/S,len(novel_mols)/S


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children
