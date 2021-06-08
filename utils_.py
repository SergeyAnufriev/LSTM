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


def sample_(model,temperature,dataset,device):
    '''Input:
        1) y_t unnormilised logits for the next symbols
        2) temperature - sampling temperature
        Return:
            next token
        https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/sampling.html'''
    hidden = model.init_hidden_(1, device)
    x = torch.tensor([dataset.dict_['G']], dtype=torch.long, device=device).unsqueeze(0)
    sequence = []
    for i in range(100):
        logits, hidden = model(x,hidden)
        prob           = softmax(logits / temperature, dim=-1)
        x              = torch.multinomial(prob, 1)[0].unsqueeze(0)
        sequence.append(x[0][0])



