import os
import numpy as np
import pytorch_lightning as pl
import optuna
import time
from sklearn.model_selection import train_test_split
from functools import partial

from network import GNN
from data_module import GNNDataModule, MoleculeDataset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bohb import TuneBOHB
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu

raytune_callback = TuneReportCheckpointCallback(
    metrics={
        'loss': 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')


def main():
    adenosine_star = False
    NUM_NODE_FEATURES = 3
    NUM_EDGE_FEATURES = 1  # at least 1
    max_epochs = 200
    n_samples = 100
    max_t_per_trial = 1000
    batch_size = 128

    # for prot_target_encoding choose: None or 'one-hot-encoding'
    prot_target_encoding = 'one-hot-encoding'

    if adenosine_star:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine', filename='human_adenosine_ligands',
                                  num_node_features=NUM_NODE_FEATURES, num_edge_features=NUM_EDGE_FEATURES,
                                  prot_target_encoding=prot_target_encoding)
    else:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                  num_node_features=NUM_NODE_FEATURES, num_edge_features=NUM_EDGE_FEATURES,
                                  prot_target_encoding=None)

    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)

    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]

    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 0
    }

    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    gnn_debug_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': 3e-4,  # learning rate
        'hidden': 256,  # embedding/hidden dimensions
        'layer_type': GAT,
        'n_layers': 5,
        'pool': 'GlobalAttention',
        'v2': True,
        'input_heads': 1,
        'active_layer': 'first',
        'batch_size': batch_size
        # 'batch_size': tune.choice([16,32,64,128])
    }

    # gnn_config = {
    #     'N': NUM_NODE_FEATURES,
    #     'E': NUM_EDGE_FEATURES,
    #     'lr': tune.loguniform(1e-4, 1e-1),  # learning rate
    #     'hidden': tune.choice([16, 32, 64, 128, 256, 512, 1024]),  # embedding/hidden dimensions
    #     'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
    #     'n_layers': tune.choice([2, 3, 4, 5, 6, 7]),
    #     'pool': tune.choice(['mean', 'GlobalAttention'])
    #     # 'batch_size': tune.choice([16,32,64,128])
    # }


    model = GNN(gnn_debug_config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=200)

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
