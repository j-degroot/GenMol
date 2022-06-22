from ast import Raise
import torch
import os
import typing
import numpy as np
import time
import pandas as pd
import pytorch_lightning as pl
from dataclasses import dataclass
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN, GAT, PNA
from utils.mol2fingerprint import calc_fps
from rdkit import Chem
from xgboost import XGBRegressor

def pred_xgb(smiles_list):
    model =XGBRegressor()
    model.load_model('temp_xgb.json')
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = calc_fps(mols)
    preds = model.predict(fps)
    # if len(smiles_list) < batch_size:
        # preds = np.concatenate((preds, np.zeros((batch_size-len(smiles_list),))), axis=0)
    return preds

from utils.encode_ligand import calc_fps
from rdkit import Chem
from xgboost import XGBRegressor


def pred_xgb(smiles_list,batch_size=64):
    model =XGBRegressor()
    model.load_model('temp_xgb.json')
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = calc_fps(mols)
    preds = model.predict(fps)
    if len(smiles_list) < batch_size:
        preds = np.concatenate((preds, np.zeros((batch_size-len(smiles_list),))), axis=0)
    return preds

class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None, name='GNN'):
        super(GNN, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.input_heads = config['input_heads']
        self.learning_rate = config['lr']
        num_features = config['N']
        self.edge_dim = config['E']
        self.hidden_size = config['hidden']
        self.layer_type = config['layer_type']
        self.num_layers = config['n_layers']
        self.batch_size = config['batch_size']
        dim = self.hidden_size

        # GIN and GraphSAGE do not include edge attr
        self.gnn = self.layer_type(num_features,dim, num_layers=self.num_layers, edge_dim=self.edge_dim,
                                   norm=torch.nn.BatchNorm1d(dim))
        if config['active_layer'] == 'first':
            self.last_layer = self.gnn._modules['convs'][0]
        elif config['active_layer'] == 'last':
            self.last_layer = self.gnn._modules['convs'][self.num_layers-1]

        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        else:
            raise ValueError('pool type not supported')

        self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))

        self.fc1 = Linear(dim, dim)

        if self.input_heads == 1:
            self.fc2 = Linear(dim, 1)

        elif self.input_heads == 2:
            self.second_input = config['second_input']
            if self.second_input == 'prot':
                self.fc_1 = Linear(4, dim)

            elif self.second_input == 'xgb':
                self.fc_1 = Linear(1,dim)
            elif self.second_input == 'fps':
                self.fc_1 = Linear(2067,dim)

            self.fc2 = Linear(2*dim, 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        x, edge_index, edge_attr, batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=0.5, training=self.training)

        if self.input_heads == 2:
            if self.second_input == 'prot':
                p = self.fc_1(graphs.p)
                x = torch.concat((x, p), dim=-1)  # on PyTorch 1.9 use torch.ca

            elif self.second_input == 'xgb':
                p = graphs.xgb_pred
                p = p.reshape(p.shape[0],1)
                p = self.fc_1(p)
                p = F.dropout(p, p=0.5, training=self.training)
                x = torch.cat((x, p), dim=1)
            elif self.second_input == 'fps':
                p = graphs.fps.float()
                p = self.fc_1(p)
                x = torch.cat((x, p), dim=1)

        x = self.fc2(x)
        return x

    def mse_loss(self, prediction, target):
        # prediction = prediction.reshape(target.shape)
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        prediction = self.forward(train_batch)
        loss = self.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss, self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = self.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        prediction = self.forward(test_batch)
        loss = self.mse_loss(prediction, test_batch.y)
        self.log('test_loss', loss, self.batch_size)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_bottleneck(self):
        return self.emb_f
