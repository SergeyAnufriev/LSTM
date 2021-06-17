from rdkit import Chem

def substructure(mol,fragment):
    return mol.GetSubstructMatches(Chem.MolFromSmiles(fragment))


