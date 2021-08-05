from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors

def substructure(s,fragment='CC(C)=O'):
    try:
      mol = Chem.MolFromSmiles(s)
      l = mol.GetSubstructMatches(Chem.MolFromSmiles(fragment))
      if len(l)>0:
        return 1
      else:
        return 0
    except:
      return 0

def valid_(smiles):
  try:
      Chem.SanitizeMol(Chem.MolFromSmiles(smiles))
      return 1
  except:
    return 0


def top_k_smiles(smiles,func,k=64):
    '''Input
    1) Smiles - list of smiles strings
    2) func - function which assigns score to a smiles string
    3) k - top K smiles by arg max func (smiles)
    Return: top K smiles list'''

    R = np.array([func(x) for x in smiles])
    indx = np.argsort(R)[-k:]
    return [smiles[x] for x in indx]


def log_p(x):

    '''Input SMILES string
    return LogP value'''
    if len(x) > 30:
        return -10**3
    else:
        try:
            mol = Chem.MolFromSmiles(x)
            lp  = Descriptors.MolLogP(mol)
            return lp
        except:
            return -10**3





smiles_l = ['CO','CCC']

print([log_p(x) for x in smiles_l])

print(top_k_smiles(smiles_l,log_p,2))





