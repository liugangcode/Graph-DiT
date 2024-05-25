from diffusion.distributions import DistributionNodes
import utils as utils
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None

    def prepare_data(self, datasets) -> None:
        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers
        self.dataloaders = {split: DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle='debug' not in self.cfg.general.name)
                            for split, dataset in datasets.items()}

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = 5
            break

        d = torch.Tensor(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index.shape[1]
                num_non_edges = all_pairs - num_edges

                data_edge_attr = torch.nn.functional.one_hot(data.edge_attr, num_classes=5).float()
                edge_types = data_edge_attr.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected
        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]
                for atom in range(n):
                    data_edge_attr = torch.nn.functional.one_hot(data.edge_attr, num_classes=5).float()
                    edges = data_edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule):
        example_batch = datamodule.example_batch()
        example_batch_x = torch.nn.functional.one_hot(example_batch.x, num_classes=118).float()[:, self.active_index]
        example_batch_edge_attr = torch.nn.functional.one_hot(example_batch.edge_attr, num_classes=5).float()

        self.input_dims = {'X': example_batch_x.size(1), 
                           'E': example_batch_edge_attr.size(1), 
                           'y': example_batch['y'].size(1)}
        self.output_dims = {'X': example_batch_x.size(1),
                            'E': example_batch_edge_attr.size(1),
                            'y': example_batch['y'].size(1)}