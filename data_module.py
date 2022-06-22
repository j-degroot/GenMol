import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils.encode_ligand import from_smiles
from utils.encode_protein import prot_target2array, prot_target2one_hot

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from rdkit import Chem
from utils.mol2fingerprint import calc_fps

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


def add_xgbpred_col(df, model_name):
    model = XGBRegressor()
    model.load_model('xgb_models/xgb_{}.json'.format(model_name))
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    fps = calc_fps(mols)
    preds = model.predict(fps)
    df['xgb_pred'] = preds
    # if len(smiles_list) < batch_size:
    # preds = np.concatenate((preds, np.zeros((batch_size-len(smiles_list),))), axis=0)
    return df


# https://github.com/deepfindr/gnn-project/blob/main/dataset_featurizer.py
class MoleculeDataset(Dataset):
    def __init__(self, root, filename, num_node_features=5, num_edge_features=1, prot_target_encoding=None, test=False,
                 transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        self.node_features = num_node_features
        self.edge_features = num_edge_features
        self.prot_target_encoding = prot_target_encoding
        self.xgb = xgb
        self.include_fps = include_fps
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'N{self.node_features}_E{self.edge_features}/data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'N{self.node_features}_E{self.edge_features}/data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        self.data = add_xgbpred_col(self.data, self.xgb)
        if not os.path.exists(os.path.join(self.processed_dir, f'N{self.node_features}_E{self.edge_features}/')):
            os.mkdir(os.path.join(self.processed_dir, f'N{self.node_features}_E{self.edge_features}/'))
        # self.data = self.data.head(1000)
        if self.prot_target_encoding != None:
            prot_target_encoder = self._target_encoder(self.prot_target_encoding)

        # self.data = self.data.head(400)  # for debugging purposes
        self.length = self.data.shape[0]
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule into PyG graph object Data
            data = from_smiles(mol['SMILES'], num_node_features=self.node_features,
                               num_edge_features=self.edge_features)
            # wrapping the value in a list to have the same shape as the network prediction
            data.y = torch.tensor([[mol['pchembl_value_Mean']]])
            if self.prot_target_encoding != None:
                data.p = prot_target_encoder(mol['target_id'])
            data.xgb_pred = torch.tensor(mol['xgb_pred'])
            if self.include_fps:
                data.fps = torch.tensor(calc_fps([mol['SMILES']]))
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'N{self.node_features}_E{self.edge_features}/data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

                f'N{self.node_features}_E{self.edge_features}/data_{index}.pt'))

        def _target_encoder(self, target_encoding):
            if target_encoding == 'one-hot-encoding':
                return prot_target2one_hot

        def len(self):
            return self.data.shape[0]

        def get(self, idx):
            """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
            if self.test:
                data = torch.load(os.path.join(self.processed_dir,
                                               f'N{self.node_features}_E{self.edge_features}/data_test_{idx}.pt'))
            else:
                data = torch.load(os.path.join(self.processed_dir,
                                               f'N{self.node_features}_E{self.edge_features}/data_{idx}.pt'))
            return data

    # Dataloader class
    # Source: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    class GNNDataModule(pl.LightningDataModule):
        def __init__(self, config, data_train, data_test, val_size=0.1):
            super().__init__()
            self.prepare_data_per_node = True
            self.batch_size = config['batch_size']
            self.num_workers = config['num_workers']
            data_train, data_val = train_test_split(data_train, test_size=val_size, random_state=0)
            self.train_data = data_train
            self.val_data = data_val
            self.test_data = data_test

        def train_dataloader(self) -> TRAIN_DATALOADERS:
            train_dataloader = DataLoader(dataset=self.train_data,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers)
            return train_dataloader

        def val_dataloader(self) -> EVAL_DATALOADERS:
            val_dataloader = DataLoader(dataset=self.val_data,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers)
            return val_dataloader

        def test_dataloader(self) -> EVAL_DATALOADERS:
            test_dataloader = DataLoader(dataset=self.test_data,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=self.num_workers)
            return test_dataloader

        def all_dataloader(self) -> EVAL_DATALOADERS:
            train_dev_sets = torch.utils.data.ConcatDataset([self.train_data, self.test_data, self.val_data])
            all_dataloader = DataLoader(dataset=train_dev_sets,  # make this legnth the length of the dataset
                                        batch_size=train_dev_sets.__len__(),
                                        shuffle=False,
                                        num_workers=self.num_workers)
            return all_dataloader

    def create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size, random_state=0, prot_enc=None,
                                                  include_fps=False):
        no_a2a = '_no_a2a' if no_a2a else ''

        p_dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine{}'.format(no_a2a),
                                    filename='human_adenosine{}_ligands'.format(no_a2a),
                                    prot_target_encoding=prot_enc, xgb='adenosine_noa2a', include_fps=include_fps)

        f_dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                    prot_target_encoding=prot_enc, xgb='a2a', include_fps=include_fps)
        all_train = []
        all_test = []
        for dataset in [p_dataset, f_dataset]:
            train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=train_size,
                                                           random_state=random_state)
            data_train = dataset[train_indices.tolist()]
            data_test = dataset[test_indices.tolist()]
            all_train.append(data_train), all_test.append(data_test)
        p_data_train = all_train[0]
        p_data_test = all_test[0]
        f_data_train = all_train[1]
        f_data_test = all_test[1]

        datamodule_config = {
            'batch_size': batch_size,
            'num_workers': 0
        }
        pre_data_module = GNNDataModule(datamodule_config, p_data_train, p_data_test)
        fine_data_module = GNNDataModule(datamodule_config, f_data_train, f_data_test)
        return pre_data_module, fine_data_module
