import torch

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

    return l
