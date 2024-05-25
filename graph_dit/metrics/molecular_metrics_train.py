import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import torch.nn as nn

class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)

        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class AtomCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, active_atoms):
        metrics_list = []
        
        for i, atom_type in enumerate(active_atoms):
            metrics_list.append(type(f'{atom_type}_CE', (AtomCE,), {})(i))

        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR])


class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        active_atoms = dataset_infos.active_atoms
        self.train_atom_metrics = AtomMetricsCE(active_atoms=active_atoms)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, log=True):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = round(val.item(),4)
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = round(val.item(),4)

        if log:
            print(f"Epoch {current_epoch}: {epoch_atom_metrics} -- {epoch_bond_metrics}")

