
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors as desc
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit import DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem.GraphDescriptors import BertzCT


# ADAPTED FROM PYTORCH-GEOMETRIC.UTILS.FROM_SMILES ---------------------------------------------------------------------
# ADDED NUM_NODES AND NUM_FEATURES ARGUMENT
x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def from_smiles(smiles: str, num_node_features: int, num_edge_features: int, with_hydrogen: bool = False,
                kekulize: bool = False):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        num_node_features:
        num_edge_features:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        x = x[:num_node_features]
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.float).view(-1, num_node_features)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
        e = e[:num_edge_features]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)  # (2, |E|)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, num_edge_features)  # (|E|, len(e))

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


# BELOW IS FROM DRUGEX -------------------------------------------------------------------------------------------------
class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                          # 'SA': sascorer.calculateScore,  # no working sascorer, left out for now
                          'Bertz': BertzCT}

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores


def calc_fps(mols, radius=3, bit_len=2048):
    ecfp = calc_ecfp(mols, radius=radius, bit_len=bit_len)
    phch = calc_physchem(mols)
    fps = np.concatenate([ecfp, phch], axis=1)
    return fps


def calc_ecfp(mols, radius=3, bit_len=2048):
    fps = np.zeros((len(mols), bit_len))
    for i, mol in enumerate(mols):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
        except:
            pass
    return fps


def calc_physchem(mols):
    prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                 'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                 'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
    fps = np.zeros((len(mols), 19))
    props = Property()
    for i, prop in enumerate(prop_list):
        props.prop = prop
        fps[:, i] = props(mols)
    return fps
# ABOVE IS FROM DRUGEX -------------------------------------------------------------------------------------------------
