### packages for visualization
from analysis.rdkit_functions import compute_molecular_metrics
from mini_moses.metrics.metrics import compute_intermediate_statistics
from metrics.property_metric import TaskModel

import torch
import torch.nn as nn

import os
import csv
import time

def result_to_csv(path, dict_data):
    file_exists = os.path.exists(path)
    log_name = dict_data.pop("log_name", None)
    if log_name is None:
        raise ValueError("The provided dictionary must contain a 'log_name' key.")
    field_names = ["log_name"] + list(dict_data.keys())
    dict_data["log_name"] = log_name
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict_data)


class SamplingMolecularMetrics(nn.Module):
    def __init__(
        self,
        dataset_infos,
        train_smiles,
        reference_smiles,
        n_jobs=1,
        device="cpu",
        batch_size=512,
    ):
        super().__init__()
        self.task_name = dataset_infos.task
        self.dataset_infos = dataset_infos
        self.active_atoms = dataset_infos.active_atoms
        self.train_smiles = train_smiles

        if reference_smiles is not None:
            print(
                f"--- Computing intermediate statistics for training for #{len(reference_smiles)} smiles ---"
            )
            start_time = time.time()
            self.stat_ref = compute_intermediate_statistics(
                reference_smiles, n_jobs=n_jobs, device=device, batch_size=batch_size
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"--- End computing intermediate statistics: using {elapsed_time:.2f}s ---"
            )
        else:
            self.stat_ref = None
    
        self.comput_config = {
            "n_jobs": n_jobs,
            "device": device,
            "batch_size": batch_size,
        }

        self.task_evaluator = {'meta_taskname': dataset_infos.task, 'sas': None, 'scs': None}
        for cur_task in dataset_infos.task.split("-")[:]:
            # print('loading evaluator for task', cur_task)
            model_path = os.path.join(
                dataset_infos.base_path, "data/evaluator", f"{cur_task}.joblib"
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            evaluator = TaskModel(model_path, cur_task)
            self.task_evaluator[cur_task] = evaluator

    def forward(self, molecules, targets, name, current_epoch, val_counter, test=False):
        if isinstance(targets, list):
            targets_cat = torch.cat(targets, dim=0)
            targets_np = targets_cat.detach().cpu().numpy()
        else:
            targets_np = targets.detach().cpu().numpy()

        unique_smiles, all_smiles, all_metrics, targets_log = compute_molecular_metrics(
            molecules,
            targets_np,
            self.train_smiles,
            self.stat_ref,
            self.dataset_infos,
            self.task_evaluator,
            self.comput_config,
        )

        if test:
            file_name = "final_smiles.txt"
            with open(file_name, "w") as fp:
                all_tasks_name = list(self.task_evaluator.keys())
                all_tasks_name = all_tasks_name.copy()
                if 'meta_taskname' in all_tasks_name:
                    all_tasks_name.remove('meta_taskname')
                if 'scs' in all_tasks_name:
                    all_tasks_name.remove('scs')

                all_tasks_str = "smiles, " + ", ".join([f"input_{task}" for task in all_tasks_name] + [f"output_{task}" for task in all_tasks_name])
                fp.write(all_tasks_str + "\n")
                for i, smiles in enumerate(all_smiles):
                    if targets_log is not None:
                        all_result_str = f"{smiles}, " + ", ".join([f"{targets_log['input_'+task][i]}" for task in all_tasks_name] + [f"{targets_log['output_'+task][i]}" for task in all_tasks_name])
                        fp.write(all_result_str + "\n")
                    else:
                        fp.write("%s\n" % smiles)
                print("All smiles saved")
        else:
            result_path = os.path.join(os.getcwd(), f"graphs/{name}")
            os.makedirs(result_path, exist_ok=True)
            text_path = os.path.join(
                result_path,
                f"valid_unique_molecules_e{current_epoch}_b{val_counter}.txt",
            )
            textfile = open(text_path, "w")
            for smiles in unique_smiles:
                textfile.write(smiles + "\n")
            textfile.close()

        all_logs = all_metrics
        if test:
            all_logs["log_name"] = "test"
        else:
            all_logs["log_name"] = (
                "epoch" + str(current_epoch) + "_batch" + str(val_counter)
            )
        
        result_to_csv("output.csv", all_logs)
        return all_smiles

    def reset(self):
        pass

if __name__ == "__main__":
    pass