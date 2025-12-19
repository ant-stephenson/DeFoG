import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import subgraph
from torch_geometric.datasets import QM7b

import utils as utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import (
    mol2smiles,
    build_molecule_with_partial_charges,
)
from analysis.rdkit_functions import compute_molecular_metrics


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return torch.zeros((1, 0), dtype=torch.float)
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return data.y[..., 3].unsqueeze(1)
        data.y = data.y[..., 3].unsqueeze(1)
        return data


class SelectHOMOTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return data.y[..., 5].unsqueeze(1)
        data.y = data.y[..., 5].unsqueeze(1)
        return data


class SelectBothTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return torch.hstack([data.y[..., 3], data.y[..., 5]]).unsqueeze(0)
        data.y = torch.hstack([data.y[..., 3], data.y[..., 5]]).unsqueeze(0)
        return data


class QM7bDataset(QM7b):
    # raw_url = (
    #     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
    #     "molnet_publish/qm7b.zip"
    # )
    # raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    # processed_url = "https://data.pyg.org/datasets/qm7b.zip"

    def __init__(
        self,
        stage,
        root,
        remove_h: bool,
        aromatic: bool,
        target_prop=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=True,
    ):
        self.target_prop = target_prop
        self.stage = stage
        self.aromatic = aromatic
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(
            root, transform, pre_transform, pre_filter
        )  # , force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    # @property
    # def raw_file_names(self):
    #     return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ["proc_tr_no_h.pt", "proc_val_no_h.pt", "proc_test_no_h.pt"]
        else:
            return ["proc_tr_h.pt", "proc_val_h.pt", "proc_test_h.pt"]

    # def download(self):
    #     """
    #     Download raw qm7b files. Taken from PyG QM7b class
    #     """
    #     # try:
    #     #     import rdkit  # noqa

    #     #     file_path = download_url(self.raw_url, self.raw_dir)
    #     #     extract_zip(file_path, self.raw_dir)
    #     #     os.unlink(file_path)

    #     #     file_path = download_url(self.raw_url2, self.raw_dir)
    #     #     os.rename(
    #     #         osp.join(self.raw_dir, "3195404"),
    #     #         osp.join(self.raw_dir, "uncharacterized.txt"),
    #     #     )
    #     # except ImportError:
    #     #     path = download_url(self.processed_url, self.raw_dir)
    #     #     extract_zip(path, self.raw_dir)
    #     #     os.unlink(path)

    #     # if files_exist(self.split_paths):
    #     #     return
    #     dataset = QM7b(
    #         root=f"/data/qm7b",
    #         force_reload=True,
    #     )

    #     dataset = pd.read_csv(self.raw_paths[1])

    #     n_samples = len(dataset)
    #     n_train = 100000
    #     n_test = int(0.1 * n_samples)
    #     n_val = n_samples - (n_train + n_test)

    #     # Shuffle dataset with df.sample, then split
    #     train, val, test = np.split(
    #         dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
    #     )

    #     train.to_csv(os.path.join(self.raw_dir, "train.csv"))
    #     val.to_csv(os.path.join(self.raw_dir, "val.csv"))
    #     test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process(self):
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
            # x = F.one_hot(
            #     torch.tensor(type_idx), num_classes=len(types)
            # ).float()
            x = torch.as_tensor(np.diag(coulomb_matrix[i]))
            data = Data(
                edge_index=edge_index, edge_attr=edge_attr, y=y, x=x, idx=i
            )
            data.num_nodes = int(edge_index.max()) + 1
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    def __process(self):
        RDLogger.DisableLog("rdApp.*")

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        if self.aromatic:
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        else:
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}  # debug

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=["mol_id"], inplace=True)

        with open(self.raw_paths[-1], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(
            self.raw_paths[0], removeHs=False, sanitize=False
        )

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(
                torch.float
            )

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(
                torch.tensor(type_idx), num_classes=len(types)
            ).float()
            y = torch.tensor([target_df.loc[i]])
            # y = torch.zeros((1, 0), dtype=torch.float)
            # y = mol.GetProp(self.target_prop)

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(
                    to_keep,
                    edge_index,
                    edge_attr,
                    relabel_nodes=True,
                    num_nodes=len(to_keep),
                )
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class QM7bDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic

        target = getattr(cfg.general, "target", None)
        regressor = getattr(cfg.general, "conditional", None)
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = SelectBothTransform()
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": QM7bDataset(
                stage="train",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),
            "val": QM7bDataset(
                stage="val",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),
            "test": QM7bDataset(
                stage="test",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),
        }
        self.test_labels = transform(datasets["test"].data, return_y=True)

        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)
        print(
            f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}"
        )
        super().__init__(cfg, datasets)


class QM7bInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic
        self.need_to_strip = False  # to indicate whether we need to ignore one output from the model
        self.compute_fcd = cfg.dataset.compute_fcd

        # if cfg.general.conditional:
        #     self.test_labels = datasets["test"].data.y

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

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt("n_counts.txt", self.n_nodes.numpy())
            self.node_types = datamodule.node_types()  # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt("atom_types.txt", self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt("edge_types.txt", self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt("valencies.txt", valencies.numpy())
            self.valency_distribution = valencies
            assert False
