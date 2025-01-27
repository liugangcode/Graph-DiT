import sys
sys.path.append('../') 

import os
import os.path as osp
import pathlib
import json

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import utils as utils
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from diffusion.distributions import DistributionNodes

bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.task = cfg.dataset.task_name
        super().__init__(cfg)

    def prepare_data(self) -> None:
        target = getattr(self.cfg.dataset, 'guidance_target', None)
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        self.root_path = root_path

        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers
        pin_memory = self.cfg.dataset.pin_memory

        dataset = Dataset(source=self.task, root=root_path, target_prop=target, transform=None)

        if len(self.task.split('-')) == 2:
            train_index, val_index, test_index, unlabeled_index = self.fixed_split(dataset)
        else:
            train_index, val_index, test_index, unlabeled_index = self.random_data_split(dataset)

        self.train_index, self.val_index, self.test_index, self.unlabeled_index = train_index, val_index, test_index, unlabeled_index
        train_index, val_index, test_index, unlabeled_index = torch.LongTensor(train_index), torch.LongTensor(val_index), torch.LongTensor(test_index), torch.LongTensor(unlabeled_index)
        if len(unlabeled_index) > 0:
            train_index = torch.cat([train_index, unlabeled_index], dim=0)
        
        train_dataset, val_dataset, test_dataset = dataset[train_index], dataset[val_index], dataset[test_index]
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)

        training_iterations = len(train_dataset) // batch_size
        self.training_iterations = training_iterations
    
    def random_data_split(self, dataset):
        nan_count = torch.isnan(dataset.y[:, 0]).sum().item()
        labeled_len = len(dataset) - nan_count
        full_idx = list(range(labeled_len))
        train_ratio, valid_ratio, test_ratio = 0.6, 0.2, 0.2
        train_index, test_index, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
        train_index, val_index, _, _ = train_test_split(train_index, train_index, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
        unlabeled_index = list(range(labeled_len, len(dataset)))
        print(self.task, ' dataset len', len(dataset), 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index), 'unlabeled len', len(unlabeled_index))
        return train_index, val_index, test_index, unlabeled_index
    
    def fixed_split(self, dataset):
        if self.task == 'O2-N2':
            test_index = [42,43,92,122,197,198,251,254,257,355,511,512,549,602,603,604]
        else:
            raise ValueError('Invalid task name: {}'.format(self.task))
        full_idx = list(range(len(dataset)))
        full_idx = list(set(full_idx) - set(test_index))
        train_ratio = 0.8
        train_index, val_index, _, _ = train_test_split(full_idx, full_idx, test_size=1-train_ratio, random_state=42)
        print(self.task, ' dataset len', len(dataset), 'train len', len(train_index), 'val len', len(val_index), 'test len', len(test_index))
        return train_index, val_index, test_index, []

    def get_train_smiles(self):
        filename = f'{self.task}.csv.gz'
        df = pd.read_csv(f'{self.root_path}/raw/{filename}')
        df_test = df.iloc[self.test_index]
        df = df.iloc[self.train_index]
        smiles_list = df['smiles'].tolist()
        smiles_list_test = df_test['smiles'].tolist()
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles_list]
        smiles_list_test = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles_list_test]
        return smiles_list, smiles_list_test
    
    def get_data_split(self):
        filename = f'{self.task}.csv.gz'
        df = pd.read_csv(f'{self.root_path}/raw/{filename}')
        df_val = df.iloc[self.val_index]
        df_test = df.iloc[self.test_index]
        df_train = df.iloc[self.train_index]
        return df_train, df_val, df_test

    def example_batch(self):
        return next(iter(self.val_loader))
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader


class Dataset(InMemoryDataset):
    def __init__(self, source, root, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.target_prop = target_prop
        self.source = source
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.source}.csv.gz']
    
    @property
    def processed_file_names(self):
        return [f'{self.source}.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        data_df = pd.read_csv(data_path)
       
        def mol_to_graph(mol, sa, sc, target, target2=None, target3=None, valid_atoms=None):
            type_idx = []
            heavy_atom_indices, active_atoms = [], []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 1:
                    type_idx.append(119-2) if atom.GetSymbol() == '*' else type_idx.append(atom.GetAtomicNum()-2)
                    heavy_atom_indices.append(atom.GetIdx())
                    active_atoms.append(atom.GetSymbol())
                    if valid_atoms is not None:
                        if not atom.GetSymbol() in valid_atoms:
                            return None, None
            x = torch.LongTensor(type_idx)

            edges_list = []
            edge_type = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if start in heavy_atom_indices and end in heavy_atom_indices:
                    start_new, end_new = heavy_atom_indices.index(start), heavy_atom_indices.index(end)
                    edges_list.append((start_new, end_new))
                    edge_type.append(bonds[bond.GetBondType()])
                    edges_list.append((end_new, start_new))
                    edge_type.append(bonds[bond.GetBondType()])
            edge_index = torch.tensor(edges_list, dtype=torch.long).t()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = edge_type

            if target3 is not None:
                y = torch.tensor([sa, sc, target, target2, target3], dtype=torch.float).view(1,-1)
            elif target2 is not None:
                y = torch.tensor([sa, sc, target, target2], dtype=torch.float).view(1,-1)
            else:
                y = torch.tensor([sa, sc, target], dtype=torch.float).view(1,-1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            return data, active_atoms
        
        # Loop through every row in the DataFrame and apply the function
        data_list = []
        len_data = len(data_df)
        with tqdm(total=len_data) as pbar:
            # --- data processing start ---
            active_atoms = set()
            for i, (sms, df_row) in enumerate(data_df.iterrows()):
                if i == sms:
                    sms = df_row['smiles']
                mol = Chem.MolFromSmiles(sms, sanitize=False)
                if len(self.target_prop.split('-')) == 2:
                    target1, target2 = self.target_prop.split('-')
                    data, cur_active_atoms = mol_to_graph(mol, df_row['SA'], df_row['SC'], df_row[target1], target2=df_row[target2])
                elif len(self.target_prop.split('-')) == 3:
                    target1, target2, target3 = self.target_prop.split('-')
                    data, cur_active_atoms = mol_to_graph(mol, df_row['SA'], df_row['SC'], df_row[target1], target2=df_row[target2], target3=df_row[target3])
                else:
                    data, cur_active_atoms = mol_to_graph(mol, df_row['SA'], df_row['SC'], df_row[self.target_prop])
                active_atoms.update(cur_active_atoms)
                data_list.append(data)
                pbar.update(1)

        torch.save(self.collate(data_list), self.processed_paths[0])


class DataInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        tasktype_dict = {
            'hiv_b': 'classification',
            'bace_b': 'classification',
            'bbbp_b': 'classification',
            'O2': 'regression',
            'N2': 'regression',
            'CO2': 'regression',
        }
        task_name = cfg.dataset.task_name
        self.task = task_name
        self.task_type = tasktype_dict.get(task_name, "regression")
        self.ensure_connected = cfg.model.ensure_connected

        datadir = cfg.dataset.datadir

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        meta_filename = os.path.join(base_path, datadir, 'raw', f'{task_name}.meta.json')
        data_root = os.path.join(base_path, datadir, 'raw')
        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                meta_dict = json.load(f)
        else:
            meta_dict = compute_meta(data_root, task_name, datamodule.train_index, datamodule.test_index)

        self.base_path = base_path
        self.active_atoms = meta_dict['active_atoms']
        self.max_n_nodes = meta_dict['max_node']
        self.original_max_n_nodes = meta_dict['max_node']
        self.n_nodes = torch.Tensor(meta_dict['n_atoms_per_mol_dist'])
        self.edge_types = torch.Tensor(meta_dict['bond_type_dist'])
        self.transition_E = torch.Tensor(meta_dict['transition_E'])

        self.atom_decoder = meta_dict['active_atoms']
        node_types = torch.Tensor(meta_dict['atom_type_dist'])
        active_index = (node_types > 0).nonzero().squeeze()
        self.node_types = torch.Tensor(meta_dict['atom_type_dist'])[active_index]
        self.nodes_dist = DistributionNodes(self.n_nodes)
        self.active_index = active_index

        val_len = 3 * self.original_max_n_nodes - 2
        meta_val = torch.Tensor(meta_dict['valencies'])
        self.valency_distribution = torch.zeros(val_len)
        val_len = min(val_len, len(meta_val))
        self.valency_distribution[:val_len] = meta_val[:val_len]
        self.y_prior = None
        self.train_ymin = []
        self.train_ymax = []


def compute_meta(root, source_name, train_index, test_index):
    pt = Chem.GetPeriodicTable()
    atom_name_list = []
    atom_count_list = []
    for i in range(2, 119):
        atom_name_list.append(pt.GetElementSymbol(i))
        atom_count_list.append(0)
    atom_name_list.append('*')
    atom_count_list.append(0)
    n_atoms_per_mol = [0] * 500
    bond_count_list = [0, 0, 0, 0, 0]
    bond_type_to_index =  {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
    valencies = [0] * 500
    tansition_E = np.zeros((118, 118, 5))
    
    filename = f'{source_name}.csv.gz'
    df = pd.read_csv(f'{root}/{filename}')
    all_index = list(range(len(df)))
    non_test_index = list(set(all_index) - set(test_index))
    df = df.iloc[non_test_index]
    tot_smiles = df['smiles'].tolist()

    n_atom_list = []
    n_bond_list = []
    for i, sms in enumerate(tot_smiles):
        try:
            mol = Chem.MolFromSmiles(sms)
        except:
            continue

        n_atom = mol.GetNumHeavyAtoms()
        n_bond = mol.GetNumBonds()
        n_atom_list.append(n_atom)
        n_bond_list.append(n_bond)

        n_atoms_per_mol[n_atom] += 1
        cur_atom_count_arr = np.zeros(118)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'H':
                continue
            elif symbol == '*':
                atom_count_list[-1] += 1
                cur_atom_count_arr[-1] += 1
            else:
                atom_count_list[atom.GetAtomicNum()-2] += 1
                cur_atom_count_arr[atom.GetAtomicNum()-2] += 1
                try:
                    valencies[int(atom.GetExplicitValence())] += 1
                except:
                    print('src', source_name,'int(atom.GetExplicitValence())', int(atom.GetExplicitValence()))
        
        tansition_E_temp = np.zeros((118, 118, 5))
        for bond in mol.GetBonds():
            start_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
            if start_atom.GetSymbol() == 'H' or end_atom.GetSymbol() == 'H':
                continue
            
            if start_atom.GetSymbol() == '*':
                start_index = 117
            else:
                start_index = start_atom.GetAtomicNum() - 2
            if end_atom.GetSymbol() == '*':
                end_index = 117
            else:
                end_index = end_atom.GetAtomicNum() - 2

            bond_type = bond.GetBondType()
            bond_index = bond_type_to_index[bond_type]
            bond_count_list[bond_index] += 2

            tansition_E[start_index, end_index, bond_index] += 2
            tansition_E[end_index, start_index, bond_index] += 2
            tansition_E_temp[start_index, end_index, bond_index] += 2
            tansition_E_temp[end_index, start_index, bond_index] += 2

        bond_count_list[0] += n_atom * (n_atom - 1) - n_bond * 2
        cur_tot_bond = cur_atom_count_arr.reshape(-1,1) * cur_atom_count_arr.reshape(1,-1) * 2 # 118 * 118
        cur_tot_bond = cur_tot_bond - np.diag(cur_atom_count_arr) * 2 # 118 * 118
        tansition_E[:, :, 0] += cur_tot_bond - tansition_E_temp.sum(axis=-1)
        assert (cur_tot_bond > tansition_E_temp.sum(axis=-1)).sum() >= 0, f'i:{i}, sms:{sms}'
    
    n_atoms_per_mol = np.array(n_atoms_per_mol) / np.sum(n_atoms_per_mol)
    n_atoms_per_mol = n_atoms_per_mol.tolist()[:51]

    atom_count_list = np.array(atom_count_list) / np.sum(atom_count_list)
    print('processed meta info: ------', filename, '------')
    print('len atom_count_list', len(atom_count_list))
    print('len atom_name_list', len(atom_name_list))
    active_atoms = np.array(atom_name_list)[atom_count_list > 0]
    active_atoms = active_atoms.tolist()
    atom_count_list = atom_count_list.tolist()

    bond_count_list = np.array(bond_count_list) / np.sum(bond_count_list)
    bond_count_list = bond_count_list.tolist()
    valencies = np.array(valencies) / np.sum(valencies)
    valencies = valencies.tolist()

    no_edge = np.sum(tansition_E, axis=-1) == 0
    first_elt = tansition_E[:, :, 0]
    first_elt[no_edge] = 1
    tansition_E[:, :, 0] = first_elt

    tansition_E = tansition_E / np.sum(tansition_E, axis=-1, keepdims=True)
    
    meta_dict = {
        'source': source_name, 
        'num_graph': len(n_atom_list), 
        'n_atoms_per_mol_dist': n_atoms_per_mol,
        'max_node': max(n_atom_list), 
        'max_bond': max(n_bond_list), 
        'atom_type_dist': atom_count_list,
        'bond_type_dist': bond_count_list,
        'valencies': valencies,
        'active_atoms': active_atoms,
        'num_atom_type': len(active_atoms),
        'transition_E': tansition_E.tolist(),
        }

    with open(f'{root}/{source_name}.meta.json', "w") as f:
        json.dump(meta_dict, f)
    
    return meta_dict


if __name__ == "__main__":
    pass