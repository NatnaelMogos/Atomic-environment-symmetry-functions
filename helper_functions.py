import numpy as np
from math import pi
import itertools
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Set, Dict


def xyz_from_file(file: str, skip_lines: int = 2) -> List[List[str]]:
    '''
    Returns a xyz block as a list. Each element of the list contains a list 
    holding the atom type and 3D coordinates
    
    param file: Path to xyz file
    param skip_lines: Number of lines to skip, by convention 2
    
    '''
    
    with open(file) as f:
        xyz_file = f.read()
    xyz_block = xyz_file.split('\n')[skip_lines:]
    return [line.split() for line in xyz_block]

def _xyz_from_string(xyz_string: str, skip_lines: int = 2) -> List[List[str]]:
    '''
    Returns a xyz block as a list. Each element of the list contains a list 
    holding the atom type and 3D coordinates
    
    param xyz_string: String that contains the xyz data
    param skip_lines: Number of lines to skip, by convention 2
    '''
    
    xyz_block = xyz_string.split('\n')[skip_lines:]

    return [line.split() for line in xyz_block if line!='']

def conformer_generator(smile: str, n_confs: int) -> List[List[List[str]]]:
    '''
    Returns a list of n_confs conformers for a given smile,
    where each conformer is a xyz block as list holding 
    lists of atom type and 3D coordinates
    
    param smile: Smile string of a molecule
    param n_confs: Number of conformers to be generated
    '''
    
    randomSeed=42
    m = Chem.AddHs(Chem.MolFromSmiles(smile))
    AllChem.EmbedMultipleConfs(m,numConfs=n_confs,randomSeed=randomSeed)
    return [_xyz_from_string(Chem.MolToXYZBlock(m,i)) for i in range(n_confs)]

def sf_exp(at1: np.ndarray, at2: np.ndarray, eta: float = 1, R_s: float = 1):
    '''
    Compute and return symmetry function for one pair of atoms
    
    param at1: First atom (atom i)
    param at2: Second atom (atom j)
    param eta: Width of the gaussian
    param R_s: Center of the gaussian
    '''
    
    a=np.array([float(_) for _ in at1])
    b=np.array([float(_) for _ in at2])
    
    R_ij = np.linalg.norm(a-b)
    return np.exp(-eta * (R_ij-R_s)**2)

def cutoff(at_1:List[str], at_2: List[str], R_c: float=None) -> float:
    '''
    Return cutoff value between atom i and j
    
    param r_ij: Distance between atom i and j
    param R_c: Cutoff radius - Max distance to consider interaction between atom i and j
    '''
    
    a=np.array([float(_) for _ in at_1])
    b=np.array([float(_) for _ in at_2])
    r_ij = np.linalg.norm(a-b)
    
    if not R_c:
        f_c=1
    elif r_ij <= R_c:
        f_c = 0.5 * (np.cos(pi * r_ij / R_c) + 1)
    elif r_ij > R_c:
        f_c = 0
    return f_c

def get_sf_params(eta_list: List = None, R_s_list: List = None) -> List[Tuple[float]]:
    '''Compute list of eta and R_s pairs to use to calculate each SF'''
    '''
    Returns a list of all different combinations of eta and R_s
    
    param eta_list: List of eta values
    param R_s_list: List of R_s values
    '''
    
    if not eta_list:
        eta_list=[1]
    if not R_s_list:
        R_s_list=[1]
        
    assert isinstance(eta_list, list), 'eta_list must be a list not {}'.format(type(eta_list).__name__)
    assert isinstance(R_s_list, list), 'R_s_list must be a list not {}'.format(type(eta_list).__name__)
    assert all(isinstance(i, (int,float)) for i in eta_list), 'values in eta_list must be of type float or int'
    assert all(isinstance(i, (int,float)) for i in R_s_list), 'values in R_s_list must be of type float or int'
    return list(itertools.product(eta_list,R_s_list))

