from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from fcd_torch import FCD as FCDMetric
from mini_moses.metrics.metrics import FragMetric, internal_diversity
from mini_moses.metrics.utils import get_mol, mapper

import re
import time
import random
random.seed(0)
import numpy as np
from multiprocessing import Pool

import torch
from metrics.property_metric import calculateSAS

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

bd_dict_x = {'O2-N2': [5.00E+04, 1.00E-03]}
bd_dict_y = {'O2-N2': [5.00E+04/2.78E+04, 1.00E-03/2.43E-05]}

selectivity = ['O2-N2']
a_dict = {}
b_dict = {}
for prop_name in selectivity:
    x1, x2 = np.log10(bd_dict_x[prop_name][0]), np.log10(bd_dict_x[prop_name][1])
    y1, y2 = np.log10(bd_dict_y[prop_name][0]), np.log10(bd_dict_y[prop_name][1])
    a = (y1-y2)/(x1-x2)
    b = y1-a*x1
    a_dict[prop_name] = a
    b_dict[prop_name] = b

def selectivity_evaluation(gas1, gas2, prop_name):
    x = np.log10(np.array(gas1))
    y = np.log10(np.array(gas1) / np.array(gas2))
    upper = (y - (a_dict[prop_name] * x + b_dict[prop_name])) > 0
    return upper

class BasicMolecularMetrics(object):
    def __init__(self, atom_decoder, train_smiles=None, stat_ref=None, task_evaluator=None, n_jobs=8, device='cpu', batch_size=512):
        self.dataset_smiles_list = train_smiles
        self.atom_decoder = atom_decoder
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self.stat_ref = stat_ref
        self.task_evaluator = task_evaluator

    def compute_relaxed_validity(self, generated, ensure_connected):
        valid = []
        num_components = []
        all_smiles = []
        valid_mols = []
        covered_atoms = set()
        direct_valid_count = 0
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
            direct_valid_flag = True if check_mol(mol, largest_connected_comp=True) is not None else False
            if direct_valid_flag:
                direct_valid_count += 1
            if not ensure_connected:
                mol_conn, _ = correct_mol(mol, connection=True)
                mol = mol_conn if mol_conn is not None else correct_mol(mol, connection=False)[0]
            else: # ensure fully connected
                mol, _ = correct_mol(mol, connection=True)
            smiles = mol2smiles(mol)
            mol = get_mol(smiles)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None and largest_mol is not None and len(smiles) > 1 and Chem.MolFromSmiles(smiles) is not None:
                    valid_mols.append(largest_mol)
                    valid.append(smiles)
                    for atom in largest_mol.GetAtoms():
                        covered_atoms.add(atom.GetSymbol())
                    all_smiles.append(smiles)
                else:
                    all_smiles.append(None)
            except Exception as e: 
                # print(f"An error occurred: {e}")
                all_smiles.append(None)
                
        return valid, len(valid) / len(generated), direct_valid_count / len(generated), np.array(num_components), all_smiles, covered_atoms

    def evaluate(self, generated, targets, ensure_connected, active_atoms=None):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, nc_validity, num_components, all_smiles, covered_atoms = self.compute_relaxed_validity(generated, ensure_connected=ensure_connected)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0

        len_active = len(active_atoms) if active_atoms is not None else 1
        
        cover_str = f"Cover {len(covered_atoms)} ({len(covered_atoms)/len_active * 100:.2f}%) atoms: {covered_atoms}"
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}% (w/o correction: {nc_validity * 100 :.2f}%), cover {len(covered_atoms)} ({len(covered_atoms)/len_active * 100:.2f}%) atoms: {covered_atoms}")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        if validity > 0: 
            dist_metrics = {'cover_str': cover_str ,'validity': validity, 'validity_nc': nc_validity}
            unique = list(set(valid))
            close_pool = False
            if self.n_jobs != 1:
                pool = Pool(self.n_jobs)
                close_pool = True
            else:
                pool = 1
            valid_mols = mapper(pool)(get_mol, valid) 
            dist_metrics['interval_diversity'] = internal_diversity(valid_mols, pool, device=self.device)
            
            start_time = time.time()
            if self.stat_ref is not None:
                kwargs = {'n_jobs': pool, 'device': self.device, 'batch_size': self.batch_size}
                kwargs_fcd = {'n_jobs': self.n_jobs, 'device': self.device, 'batch_size': self.batch_size}
                try:
                    dist_metrics['sim/Frag'] = FragMetric(**kwargs)(gen=valid_mols, pref=self.stat_ref['Frag'])
                except:
                    print('error: ', 'pool', pool)
                    print('valid_mols: ', valid_mols)
                dist_metrics['dist/FCD'] = FCDMetric(**kwargs_fcd)(gen=valid, pref=self.stat_ref['FCD'])

            if self.task_evaluator is not None:
                evaluation_list = list(self.task_evaluator.keys())
                evaluation_list = evaluation_list.copy()

                assert 'meta_taskname' in evaluation_list
                meta_taskname = self.task_evaluator['meta_taskname']
                evaluation_list.remove('meta_taskname')
                meta_split = meta_taskname.split('-')

                valid_index = np.array([True if smiles else False for smiles in all_smiles])
                targets_log = {}
                for i, name in enumerate(evaluation_list):
                    targets_log[f'input_{name}'] = np.array([float('nan')] * len(valid_index))
                    targets_log[f'input_{name}'] = targets[:, i]
                
                targets = targets[valid_index]
                if len(meta_split) == 2:
                    cached_perm = {meta_split[0]: None, meta_split[1]: None}
                
                for i, name in enumerate(evaluation_list):
                    if name == 'scs':
                        continue
                    elif name == 'sas':
                        scores = calculateSAS(valid)
                    else:
                        scores = self.task_evaluator[name](valid)
                    targets_log[f'output_{name}'] = np.array([float('nan')] * len(valid_index))
                    targets_log[f'output_{name}'][valid_index] = scores
                    if name in ['O2', 'N2', 'CO2']:
                        if len(meta_split) == 2:
                            cached_perm[name] = scores
                        scores, cur_targets = np.log10(scores), np.log10(targets[:, i])
                        dist_metrics[f'{name}/mae'] = np.mean(np.abs(scores - cur_targets))
                    elif name == 'sas':
                        dist_metrics[f'{name}/mae'] = np.mean(np.abs(scores - targets[:, i]))
                    else:
                        true_y = targets[:, i]
                        predicted_labels = (scores >= 0.5).astype(int)
                        acc = (predicted_labels == true_y).sum() / len(true_y)
                        dist_metrics[f'{name}/acc'] = acc

                if len(meta_split) == 2:
                    if cached_perm[meta_split[0]] is not None and cached_perm[meta_split[1]] is not None:
                        task_name = self.task_evaluator['meta_taskname']
                        upper = selectivity_evaluation(cached_perm[meta_split[0]], cached_perm[meta_split[1]], task_name)
                        dist_metrics[f'selectivity/{task_name}'] = np.sum(upper)

            end_time = time.time()
            elapsed_time = end_time - start_time
            max_key_length = max(len(key) for key in dist_metrics)
            print(f'Details over {len(valid)} ({len(generated)}) valid (total) molecules, calculating metrics using {elapsed_time:.2f} s:')
            strs = ''
            for i, (key, value) in enumerate(dist_metrics.items()):
                if isinstance(value, (int, float, np.floating, np.integer)):
                    strs = strs + f'{key:>{max_key_length}}:{value:<7.4f}\t'
                if i % 4 == 3:
                    strs = strs + '\n'
            print(strs)

            if close_pool:
                pool.close()
                pool.join()
        else:
            unique = []
            dist_metrics = {}
            targets_log = None
        return unique, dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), all_smiles, dist_metrics, targets_log

def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                if len(atomid_valence) == 2:
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if verbose:
                        print("atomic num of atom with a large valence", an)
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                        # print("Formal charge added")
                else:
                    continue
    return mol

def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol, connection=False):
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        if connection:
            mol_conn = connect_fragments(mol)
            # if mol_conn is not None:
            mol = mol_conn
            if mol is None:
                return None, no_correct
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            try:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                check_idx = 0
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    type = int(b.GetBondType())
                    queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                    if type == 12:
                        check_idx += 1
                queue.sort(key=lambda tup: tup[1], reverse=True)

                if queue[-1][1] == 12:
                    return None, no_correct
                elif len(queue) > 0:
                    start = queue[check_idx][2]
                    end = queue[check_idx][3]
                    t = queue[check_idx][1] - 1
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, bond_dict[t])
            except Exception as e:
                # print(f"An error occurred in correction: {e}")
                return None, no_correct
    return mol, no_correct


def check_mol(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


##### connect fragements
def select_atom_with_available_valency(frag):
    atoms = list(frag.GetAtoms())
    random.shuffle(atoms)
    for atom in atoms:
        if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0:
            return atom

    return None

def select_atoms_with_available_valency(frag):
    return [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0]

def try_to_connect_fragments(combined_mol, frag, atom1, atom2):
    # Make copies of the molecules to try the connection
    trial_combined_mol = Chem.RWMol(combined_mol)
    trial_frag = Chem.RWMol(frag)
    
    # Add the new fragment to the combined molecule with new indices
    new_indices = {atom.GetIdx(): trial_combined_mol.AddAtom(atom) for atom in trial_frag.GetAtoms()}
    
    # Add the bond between the suitable atoms from each fragment
    trial_combined_mol.AddBond(atom1.GetIdx(), new_indices[atom2.GetIdx()], Chem.BondType.SINGLE)
    
    # Adjust the hydrogen count of the connected atoms
    for atom_idx in [atom1.GetIdx(), new_indices[atom2.GetIdx()]]:
        atom = trial_combined_mol.GetAtomWithIdx(atom_idx)
        num_h = atom.GetTotalNumHs()
        atom.SetNumExplicitHs(max(0, num_h - 1))
        
    # Add bonds for the new fragment
    for bond in trial_frag.GetBonds():
        trial_combined_mol.AddBond(new_indices[bond.GetBeginAtomIdx()], new_indices[bond.GetEndAtomIdx()], bond.GetBondType())
    
    # Convert to a Mol object and try to sanitize it
    new_mol = Chem.Mol(trial_combined_mol)
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol  # Return the new valid molecule
    except Chem.MolSanitizeException:
        return None  # If the molecule is not valid, return None

def connect_fragments(mol):
    # Get the separate fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) < 2:
        return mol

    combined_mol = Chem.RWMol(frags[0])

    for frag in frags[1:]:
        # Select all atoms with available valency from both molecules
        atoms1 = select_atoms_with_available_valency(combined_mol)
        atoms2 = select_atoms_with_available_valency(frag)
        
        # Try to connect using all combinations of available valency atoms
        for atom1 in atoms1:
            for atom2 in atoms2:
                new_mol = try_to_connect_fragments(combined_mol, frag, atom1, atom2)
                if new_mol is not None:
                    # If a valid connection is made, update the combined molecule and break
                    combined_mol = new_mol
                    break
            else:
                # Continue if the inner loop didn't break (no valid connection found for atom1)
                continue
            # Break if the inner loop did break (valid connection found)
            break
        else:
            # If no valid connections could be made with any of the atoms, return None
            return None

    return combined_mol

#### connect fragements

def compute_molecular_metrics(molecule_list, targets, train_smiles, stat_ref, dataset_info, task_evaluator, comput_config):
    """ molecule_list: (dict) """

    atom_decoder = dataset_info.atom_decoder
    active_atoms = dataset_info.active_atoms
    ensure_connected = dataset_info.ensure_connected
    metrics = BasicMolecularMetrics(atom_decoder, train_smiles, stat_ref, task_evaluator, **comput_config)
    evaluated_res = metrics.evaluate(molecule_list, targets, ensure_connected, active_atoms)
    all_smiles = evaluated_res[-3]
    all_metrics = evaluated_res[-2]
    targets_log = evaluated_res[-1]
    unique_smiles = evaluated_res[0]

    return unique_smiles, all_smiles, all_metrics, targets_log

if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    print(block_mol)
