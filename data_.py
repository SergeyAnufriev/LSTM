import pandas as pd
from torch.utils.data import Dataset
import torch


class Dataset_(Dataset):
    def __init__(self,dir_):

        self.df       = pd.read_csv(dir_,sep=" ",header=None)
        self.df['l']  = self.df.iloc[:,0].apply(lambda x:len(x))
        self.max_l    = max(self.df['l'].values)
        self.dict_    = self.pandas_to_dict()
        self.dict_inv = {value: key for key, value in self.dict_.items()}

    def __len__(self):
        return len(self.df)

    def pandas_to_dict(self):
        '''Input: pandas dataset with smiles in first column
        Output: dictionary where each unique symbol in the dataset is
        assigned with unique integer'''

        separator       = ', '
        unique_sym_list = list(set(separator.join(list(self.df.iloc[:,0]))))
        unique_sym_list += ['G','E','A']

        return {key:value for value, key in zip(range(len(unique_sym_list)),unique_sym_list) if key !=' '}

    def seq_(self,smiles):
        '''Input smiles string, e.g CCO
        Output: sequence with 'start_token'CCO'END_TOKEN'pad,pad
        in didgit form'''''

        numeric =  [self.dict_[x] for x in smiles]
        seq     =  [self.dict_['G']] + numeric
        seq     += [self.dict_['E']]
        pad_l   =   self.max_l - len(smiles)

        if pad_l > 0:
            seq += [self.dict_['A']]*pad_l

        return seq

    def msk_(self,x):
        '''Input - length of smiles string
        Output - {1,0} mask for the string,
                 which contains (0 if pad otherwise 1)'''
        pad_l   =   self.max_l - x
        if pad_l > 0:
            mask = torch.tensor([1.]*(x+1) + [0.]*pad_l,dtype=torch.float32)
        else:
            mask = torch.tensor([1.]*(x+1),dtype=torch.float32)
        return mask

    def __getitem__(self,idx):
        '''Input index in dataset:
        Output: input_seq: (one-hot) size = [bz,seq_l], dtype int
                target_seq (int) size = [bz,seq_l] dtype int
                mask: (one-hot) size = [bz,seq_l] dtype {0,1}'''

        smiles      = self.df.iloc[:,0][idx]
        seq         = self.seq_(smiles)
        target_seq_ = torch.tensor(seq[1:],dtype=torch.long)
        input_seq   = torch.tensor(seq[:-1],dtype=torch.long)


        return input_seq, target_seq_, self.msk_(len(smiles))





