import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import PandasTools
from torch import seed

import utils
from utils.encode_ligand import calc_fps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)

def main():
    try:
        df = pd.read_csv('data/a2aar/raw/human_a2aar_ligands')
        df = df[['SMILES', 'pchembl_value_Mean']]
        # df = df.dropna(axis=0)
        # df['SMILES'] = df['SMILES'].map(canonical_smiles)
        PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
        print('Processed Smiles to Mol object')
        target_values = df['pchembl_value_Mean'].to_numpy()
        fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
        print('Finished calculating fingerprints')
        print(fps.shape)
        print(target_values.shape)
        # X_train, X_test, y_train, y_test = train_test_split(fps, target_values, test_size=0.5,random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(fps, target_values, test_size=0.1,random_state=0)
        model = XGBRegressor()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        print(f"RMSE = {rmse}")
        model.save_model('xgb_models/xgb_a2a.json')
        exit()
        # params = {'max_depth': [3],
        #             'learning_rate': [0.01],
        #             'n_estimators': [100],
        #             'colsample_bylevel': [0.5]}
        params = {'booster': ['gbtree', 'gblinear', 'dart'],
                  'max_depth': [3, 6, 10],
                  'learning_rate': [0.001, 0.01, 0.3],
                  'n_estimators': [100, 500, 1000]}
        xgbr = XGBRegressor()
        regrs = GridSearchCV(cv=2, estimator=xgbr, param_grid=params, scoring='neg_mean_squared_error', verbose=1)
        regrs.fit(X_train, y_train)
        joblib.dump(regrs, 'model_result_big.pkl')
        print("Best parameters:", regrs.best_params_)
        print("Lowest MSE: ", (-regrs.best_score_))
        xgbr.save_model('xgb.json')
        #print("Lowest RMSE: ", (-regrs.best_score_))
    except Exception as e:
        print('Could not load data. \nProcessing data.', e)
        df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)
        df = df[['Smiles', 'pChEMBL_Value']]

if __name__=='__main__':
    main()