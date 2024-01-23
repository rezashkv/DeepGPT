import copy

from transformers import (
    EvalPrediction
)
from utils.light_evaluater import Evaluator
from torch_geometric.datasets import (
    MoleculeNet
)

from torch_geometric.utils import (
    index_to_mask,
)

import json
import logging
import os

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr
from torch_geometric.data.separate import separate


def prepare_splits(dataset, seed):
    """Ready train/val/test splits.

    Determine the type of split from the config and call the corresponding
    split generation / verification function.
    """
    split_mode = dataset.split_mode

    if split_mode == 'standard':
        setup_standard_split(dataset)
    elif split_mode == 'random':
        setup_random_split(dataset, seed)
    elif split_mode.startswith('cv-'):
        cv_type, k = split_mode.split('-')[1:]
        setup_cv_split(dataset, cv_type, int(k))
    elif split_mode == "fixed":
        setup_fixed_split(dataset)
    elif split_mode == "sliced":
        setup_sliced_split(dataset)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")


def setup_standard_split(dataset):
    """Select a standard split.

    Use standard splits that come with the dataset. Pick one split based on the
    ``split_index`` from the config file if multiple splits are available.

    GNNBenchmarkDatasets have splits that are not prespecified as masks. Therefore,
    they are handled differently and are first processed to generate the masks.

    Raises:
        ValueError: If any one of train/val/test mask is missing.
        IndexError: If the ``split_index`` is greater or equal to the total
            number of splits available.
    """
    split_index = dataset.split_index
    task_level = dataset.task

    if task_level == 'node':
        for split_name in 'train_mask', 'val_mask', 'test_mask':
            mask = getattr(dataset.data, split_name, None)
            # Check if the train/val/test split mask is available
            if mask is None:
                raise ValueError(f"Missing '{split_name}' for standard split")

            # Pick a specific split if multiple splits are available
            if mask.dim() == 2:
                if split_index >= mask.shape[1]:
                    raise IndexError(f"Specified split index ({split_index}) is "
                                     f"out of range of the number of available "
                                     f"splits ({mask.shape[1]}) for {split_name}")
                set_dataset_attr(dataset, split_name, mask[:, split_index],
                                 len(mask[:, split_index]))
            else:
                if split_index != 0:
                    raise IndexError(f"This dataset has single standard split")

    elif task_level == 'graph':
        for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")

    elif task_level == 'link_pred':
        for split_name in 'train_edge_index', 'val_edge_index', 'test_edge_index':
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")

    else:
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")


def setup_random_split(dataset, seed):
    """Generate random splits.

    Generate random train/val/test based on the ratios defined in the config
    file.

    Raises:
        ValueError: If the number split ratios is not equal to 3, or the ratios
            do not sum up to 1.
    """
    split_ratios = dataset.split

    if len(split_ratios) != 3:
        raise ValueError(
            f"Three split ratios is expected for train/val/test, received "
            f"{len(split_ratios)} split ratios: {repr(split_ratios)}")
    elif sum(split_ratios) != 1 and sum(split_ratios) != len(dataset):
        raise ValueError(
            f"The train/val/test split ratios must sum up to 1/length of the dataset, input ratios "
            f"sum up to {sum(split_ratios):.2f} instead: {repr(split_ratios)}")

    train_index, val_test_index = next(
        ShuffleSplit(
            train_size=split_ratios[0],
            random_state=seed
        ).split(dataset.data.y, dataset.data.y)
    )

    if isinstance(split_ratios[0], float):
        val_test_ratio = split_ratios[1] / (1 - split_ratios[0])
    else:
        val_test_ratio = split_ratios[1]

    val_index, test_index = next(
        ShuffleSplit(
            train_size=val_test_ratio,
            random_state=seed
        ).split(dataset.data.y[val_test_index], dataset.data.y[val_test_index])
    )
    val_index = val_test_index[val_index]
    test_index = val_test_index[test_index]

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def setup_fixed_split(dataset):
    """Generate fixed splits.

    Generate fixed train/val/test based on the ratios defined in the config
    file.
    """
    train_index = list(range(dataset.split[0]))
    val_index = list(range(dataset.split[0], sum(dataset.split[:2])))
    test_index = list(range(sum(dataset.split[:2]), sum(dataset.split)))

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def setup_sliced_split(dataset):
    """Generate sliced splits.

    Generate sliced train/val/test based on the ratios defined in the config
    file.
    """
    train_index = list(range(*dataset.split[0]))
    val_index = list(range(*dataset.split[1]))
    test_index = list(range(*dataset.split[2]))

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )

    task_level = dataset.task
    if task_level == 'node':
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for split_name, split_index in zip(split_names, splits):
            mask = index2mask(split_index, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_name, mask, len(mask))

    elif task_level == 'graph':
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(dataset, split_name, split_index, len(split_index))

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")


def setup_cv_split(dataset, cv_type, k):
    """Generate cross-validation splits.

    Generate `k` folds for cross-validation based on `cv_type` procedure. Save
    these to disk or load existing splits, then select particular train/val/test
    split based on cfg.dataset.split_index from the config object.

    Args:
        dataset: PyG dataset object
        cv_type: Identifier for which sklearn fold splitter to use
        k: how many cross-validation folds to split the dataset into

    Raises:
        IndexError: If the `split_index` is greater than or equal to `k`
    """
    split_index = dataset.split_index
    split_dir = dataset.split_dir

    if split_index >= k:
        raise IndexError(f"Specified split_index={split_index} is "
                         f"out of range of the number of folds k={k}")

    os.makedirs(split_dir, exist_ok=True)
    save_file = os.path.join(
        split_dir,
        f"{dataset.format}_{dataset.name}_{cv_type}-{k}.json"
    )
    if not os.path.isfile(save_file):
        create_cv_splits(dataset, cv_type, k, save_file)
    with open(save_file) as f:
        cv = json.load(f)
    assert cv['dataset'] == dataset.name, "Unexpected dataset CV splits"
    assert cv['n_samples'] == len(dataset), "Dataset length does not match"
    assert cv['n_splits'] > split_index, "Fold selection out of range"
    assert k == cv['n_splits'], f"Expected k={k}, but {cv['n_splits']} found"

    test_ids = cv[str(split_index)]
    val_ids = cv[str((split_index + 1) % k)]
    train_ids = []
    for i in range(k):
        if i != split_index and i != (split_index + 1) % k:
            train_ids.extend(cv[str(i)])

    set_dataset_splits(dataset, [train_ids, val_ids, test_ids])


def create_cv_splits(dataset, cv_type, k, file_name):
    """Create cross-validation splits and save them to file.
    """
    n_samples = len(dataset)
    if cv_type == 'stratifiedkfold':
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples), dataset.data.y)
    elif cv_type == 'kfold':
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples))
    else:
        ValueError(f"Unexpected cross-validation type: {cv_type}")

    splits = {'n_samples': n_samples,
              'n_splits': k,
              'cross_validator': kf.__str__(),
              'dataset': dataset.name
              }
    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    with open(file_name, 'w') as f:
        json.dump(splits, f)
    logging.info(f"[*] Saved newly generated CV splits by {kf} to {file_name}")


index2mask = index_to_mask  # TODO Backward compatibility


def load_pyg(name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (str): dataset name
        dataset_dir (str): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    dataset = MoleculeNet(dataset_dir, name)

    return dataset


class MoleculeNetSubset(MoleculeNet):
    def __init__(self, root: str, name: str):
        super().__init__(root, name)

class MoleculeNetDataset:
    def __init__(self, data_args, training_args):

        self.dataset = load_pyg(data_args.dataset_name, data_args.data_path)
        self.dataset.split_mode = data_args.split_mode
        self.dataset_type = data_args.dataset_type
        self.dataset.split = [0.8, 0.1, 0.1]
        self.dataset.task = "graph"
        self.n_tasks = data_args.n_tasks

        prepare_splits(self.dataset, training_args.seed)
        self.metric = data_args.metric

        if training_args.do_train:
            id = self.dataset.data['train_graph_index']
            self.train_dataset = self.dataset[id]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            id = self.dataset.data['val_graph_index']
            self.eval_dataset = self.dataset[id]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            id = self.dataset.data['test_graph_index']
            self.predict_dataset = self.dataset[id]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.data_collator = None

        if data_args.metric is not None:
            self.evaluator = Evaluator(data_args.dataset_name, data_args.metric, self.n_tasks)
        else:
            if "classification" in self.dataset_type:
                self.metric = "AUROC"
                self.evaluator = Evaluator(data_args.dataset_name, "auroc", self.n_tasks)
            else:
                self.metric = "RMSE"
                self.evaluator = Evaluator(data_args.dataset_name, "rmse", self.n_tasks,
                                           mean=self.train_dataset.mean.numpy(),
                                           std=self.train_dataset.std.numpy())

    def compute_metrics(self, p: EvalPrediction):
        y_true = p.label_ids
        y_pred = p.predictions
        return {self.metric: self.evaluator.eval(y_true, y_pred)}
