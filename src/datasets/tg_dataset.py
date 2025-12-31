import os
import pathlib
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import QM7b
from hydra.utils import get_original_cwd

from datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
    RemoveYTransform,
)


class QM7bDataset(InMemoryDataset):

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat"

    def __init__(
        self,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = "qm7b"
        root = root

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self) -> str:
        return "qm7b.mat"

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    @classmethod
    def save(cls, data_list, path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        torch.save((data.to_dict(), slices, data.__class__), path)

    def load(self, path: str, data_cls=Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = torch.load(path)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def process(self) -> None:
        from scipy.io import loadmat

        data = loadmat(self.raw_paths[0])

        n_samples = data["X"].shape[0]
        n_train = n_samples * 4 // 5
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)
        tr_idx, val_idx, test_idx = np.split(
            np.arange(n_samples), [n_train, n_val + n_train]
        )
        idx = {0: tr_idx, 1: val_idx, 2: test_idx}

        X = data["X"][idx[self.file_idx]]
        T = data["T"][idx[self.file_idx]]

        coulomb_matrix = torch.from_numpy(X)
        target = torch.from_numpy(T).to(torch.float)

        data_list = []
        for i in range(target.shape[0]):
            edge_index = (
                coulomb_matrix[i].nonzero(as_tuple=False).t().contiguous()
            )
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y = target[i].view(1, -1)
            x = coulomb_matrix[i].diag().view(1, -1)
            data = Data(
                edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, idx=i
            )
            data.num_nodes = int(edge_index.max()) + 1
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[self.file_idx])


class QM7bDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": QM7bDataset(
                root=root_path,
                transform=transform,
                split="train",
            ),
            "val": QM7bDataset(
                root=root_path,
                transform=transform,
                split="val",
            ),
            "test": QM7bDataset(
                root=root_path,
                transform=transform,
                split="test",
            ),
        }

        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)
        print(
            f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}"
        )
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset


class QM7bInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.is_molecular = True
        self.spectre = True
        self.use_charge = False
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic
        self.need_to_strip = False  # to indicate whether we need to ignore one output from the model
        self.compute_fcd = cfg.dataset.compute_fcd
        # self.datamodule = datamodule
        self.dataset_name = datamodule.inner.dataset_name
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        # self.edge_types = datamodule.edge_counts()
        self.name = "qm7b"
        if self.remove_h:
            self.atom_encoder = {"C": 0, "N": 1, "O": 2, "F": 3}
            self.atom_decoder = ["C", "N", "O", "F"]
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 9
            self.max_weight = 150
            self.n_nodes = torch.tensor(
                [
                    0,
                    2.2930e-05,
                    3.8217e-05,
                    6.8791e-05,
                    2.3695e-04,
                    9.7072e-04,
                    0.0046472,
                    0.023985,
                    0.13666,
                    0.83337,
                ]
            )
            self.node_types = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
            if self.aromatic:
                self.edge_types = torch.tensor(
                    [0.7261, 0.2384, 0.0274, 0.0081, 0.0]
                )
            else:
                self.edge_types = torch.tensor(
                    [0.7261, 0.2384, 0.0274, 0.0081]
                )  # debug

            super().complete_infos(
                n_nodes=self.n_nodes, node_types=self.node_types
            )
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor(
                [2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073]
            )
        else:
            self.atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
            self.atom_decoder = ["H", "C", "N", "O", "F"]
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.tensor(
                [
                    0,
                    0,
                    0,
                    1.5287e-05,
                    3.0574e-05,
                    3.8217e-05,
                    9.1721e-05,
                    1.5287e-04,
                    4.9682e-04,
                    1.3147e-03,
                    3.6918e-03,
                    8.0486e-03,
                    1.6732e-02,
                    3.0780e-02,
                    5.1654e-02,
                    7.8085e-02,
                    1.0566e-01,
                    1.2970e-01,
                    1.3332e-01,
                    1.3870e-01,
                    9.4802e-02,
                    1.0063e-01,
                    3.3845e-02,
                    4.8628e-02,
                    5.4421e-03,
                    1.4698e-02,
                    4.5096e-04,
                    2.7211e-03,
                    0.0000e00,
                    2.6752e-04,
                ]
            )

            self.node_types = torch.tensor(
                [0.5122, 0.3526, 0.0562, 0.0777, 0.0013]
            )
            self.edge_types = torch.tensor(
                [0.88162, 0.11062, 5.9875e-03, 1.7758e-03, 0]
            )

            if self.aromatic:
                self.edge_types = torch.tensor(
                    [0.88162, 0.11062, 5.9875e-03, 1.7758e-03, 0]
                )
            else:
                self.edge_types = torch.tensor(
                    [0.88162, 0.11062, 5.9875e-03, 1.7758e-03]
                )
            # self.edge_types = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03])  # debug

            super().complete_infos(
                n_nodes=self.n_nodes, node_types=self.node_types
            )
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor(
                [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
            )

        # if recompute_statistics:
        #     np.set_printoptions(suppress=True, precision=5)
        #     self.n_nodes = datamodule.node_counts()
        #     print("Distribution of number of nodes", self.n_nodes)
        #     np.savetxt("n_counts.txt", self.n_nodes.numpy())
        #     self.node_types = datamodule.node_types()  # There are no node types
        #     print("Distribution of node types", self.node_types)
        #     np.savetxt("atom_types.txt", self.node_types.numpy())

        #     self.edge_types = datamodule.edge_counts()
        #     print("Distribution of edge types", self.edge_types)
        #     np.savetxt("edge_types.txt", self.edge_types.numpy())

        #     valencies = datamodule.valency_count(self.max_n_nodes)
        #     print("Distribution of the valencies", valencies)
        #     np.savetxt("valencies.txt", valencies.numpy())
        #     self.valency_distribution = valencies
        #     assert False

    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        data.edge_attr = F.one_hot(
            data.edge_attr, num_classes=self.num_edge_types
        ).float()

        return data
