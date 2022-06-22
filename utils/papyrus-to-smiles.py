import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

path = 'data/Papyrus.tsv.gz'
df = pd.read_csv(path, sep='\t', compression='gzip')
df = df[['SMILES', 'Activity_class', 'target_id', 'pchembl_value_Mean', 'relation']]

# list_of_proteins = pd.read_csv('data/human_G-coupled_proteins')
# list_of_proteins = list_of_proteins['target_id']
# list_of_proteins = list_of_proteins.values.tolist()
list_of_proteins = ['P30542', 'P0DMS8', 'P29275']

# Only keep molecules with the target_ID for the receptor we want
df = df[df['target_id'].str.contains('|'.join(list_of_proteins))]

print(df.head())
print('Length: ', df.shape[0])
df.to_csv('adenosine_no-a2a_ligands')

# ligand_path = 'Papyrus.tsv.gz'
# receptor_path = 'data/protein_targets.gz'
# proteins = pd.read_csv(receptor_path, sep='\t',low_memory=False)
# ligands = pd.read_csv(ligand_path, sep='\t',low_memory=False)

# adenosine_ligands = pd.DataFrame() 

# def is_adenosine(target_ID):
#     # target_ID = str(row['target_id'])
#     protein = proteins[proteins['target_id'].str.contains(target_ID)]
#     if 'Membrane receptor->Family A G' in str(protein['Classification']):
#         return True
#     else:
#         return False

# aden_proteins = proteins[proteins['Classification'].str.contains('Membrane receptor->Family A G',na=False)]
# aden_proteins = aden_proteins[aden_proteins['Organism'].str.contains('Human',na=False)]

# aden_proteins.to_csv('human_G-coupled_proteins')
