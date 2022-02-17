import os
import time
import itertools
import numpy as np
from math import pi
from typing import List, Tuple, Set, Dict

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from helper_functions import*


def SF_from_mol(mol: List[List[str]], at_types: Set[str], eta_list: List = None, R_s_list: List = None, R_c: float=None) -> List[Dict[str, float]]:
    '''Compute atomic environment symmetry functions for all atoms in a molecule
    
    param mol: xyz block of a molecule
    param at_types: List of distinct atom types in mol
    param eta_list: List of eta values
    param R_s_list: List of R_s values
    '''
    
    sf_params = get_sf_params(eta_list, R_s_list)
    pairwise_permutations = np.asarray(list(itertools.permutations(mol,2)))
    atom_list = [{} for i in range(len(mol))] #list to store dicts with sfs for each atom (e.g.: 4 atoms in mol -> 4 dicts)

    for i in range(len(atom_list)):
        for j in range(len(pairwise_permutations)):
            if (pairwise_permutations[j][0] == mol[i]).all(): #'center atom vs. atom in xyz'
                for p1,p2 in sf_params:
                    at_dict = atom_list[i]
                    
                    f_c = cutoff(pairwise_permutations[j][0][1:],pairwise_permutations[j][1][1:])
                    sf = sf_exp(pairwise_permutations[j][0][1:],
                                pairwise_permutations[j][1][1:],
                                eta=p1,R_s=p2) * f_c
                    
                    key = "".join(pairwise_permutations[j][:,0])+str(p1)+str(p2) #'long key name to check where values come from'
                    at_dict[key] = at_dict.setdefault(key,0) + sf

            else:
                continue
    
    return atom_list

if __name__ == '__main__':
    from param_file import R_c, eta, R_s, at_types
    c_g=conformer_generator('C=O',10)
    print(*SF_from_mol(c_g[0], at_types,eta,R_s),sep='\n\n')