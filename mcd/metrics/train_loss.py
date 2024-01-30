import time
import torch
import torch.nn as nn
from metrics.abstract_metrics import CrossEntropyMetric
from torchmetrics import Metric, MeanSquaredError

# from 2:He to 119:*
valencies_check = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 6, 6, 7, 6, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 7, 6, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 7, 6, 5, 6, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 7, 6, 5, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
valencies_check = torch.tensor(valencies_check)

weight_check = [4.003, 6.941, 9.012, 10.812, 12.011, 14.007, 15.999, 18.998, 20.18, 22.99, 24.305, 26.982, 28.086, 30.974, 32.067, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.39, 69.723, 72.61, 74.922, 78.96, 79.904, 83.8, 85.468, 87.62, 88.906, 91.224, 92.906, 95.94, 98.0, 101.07, 102.906, 106.42, 107.868, 112.412, 114.818, 118.711, 121.76, 127.6, 126.904, 131.29, 132.905, 137.328, 138.906, 140.116, 140.908, 144.24, 145.0, 150.36, 151.964, 157.25, 158.925, 162.5, 164.93, 167.26, 168.934, 173.04, 174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.078, 196.967, 200.59, 204.383, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.038, 231.036, 238.029, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 281.0, 285.0, 284.0, 289.0, 288.0, 293.0, 292.0, 294.0, 294.0]
weight_check = torch.tensor(weight_check)

class AtomWeightMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_loss', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        global weight_check
        self.weight_check = weight_check

    def update(self, X, Y):
        atom_pred_num = X.argmax(dim=-1)
        atom_real_num = Y.argmax(dim=-1)
        self.weight_check = self.weight_check.type_as(X)

        pred_weight = self.weight_check[atom_pred_num]
        real_weight = self.weight_check[atom_real_num]

        lss = 0
        lss += torch.abs(pred_weight.sum(dim=-1) - real_weight.sum(dim=-1)).sum()
        self.total_loss += lss
        self.total_samples += X.size(0)

    def compute(self):
        return self.total_loss / self.total_samples


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, weight_node=None, weight_edge=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.weight_loss = AtomWeightMetric()

        self.y_loss = MeanSquaredError()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, node_mask, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        loss_weight = self.weight_loss(masked_pred_X, true_X)
        
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        
        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        return self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E + loss_weight

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time, log=True):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_weight_loss = self.weight_loss.compute() if self.weight_loss.total_samples > 0 else -1

        if log:
            print(f"Epoch {current_epoch} finished: X_CE: {epoch_node_loss :.4f} -- E_CE: {epoch_edge_loss :.4f} "
                f"Weight: {epoch_weight_loss :.4f} "
                f"-- Time taken {time.time() - start_epoch_time:.1f}s ")