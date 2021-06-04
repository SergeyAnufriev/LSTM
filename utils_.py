import torch
from torch.nn.functional import softmax

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



def test_loss(test_l,model):
    '''Input:
    1) test_l - test data loader
    2) model  - model to be tested]
    Output: average loss on the whole dataset'''

    model.eval()
    losses = 0

    for input_seq, target_seq, mask in test_l:

        pred,_,_   = model(input_seq)
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