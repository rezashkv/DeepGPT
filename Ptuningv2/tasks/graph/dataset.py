import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy as sp
import torch
from dgl import load_graphs, save_graphs
import dgl.backend as F
from dgllife.utils.io import pmap
from rdkit import Chem
from torch.utils.data import Dataset
from transformers import (
    EvalPrediction
)
from sklearn.metrics import roc_auc_score
from datasets.load import load_dataset
from utils.graphormer_utils import preprocess_item, GraphormerDataCollator
from ogb.utils.mol import smiles2graph
from utils.light_evaluater import Evaluator

logger = logging.getLogger(__name__)

SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}

class TransformerGraphDataset:

    def __init__(self, data_args, training_args) -> None:
        raw_datasets = load_dataset(data_args.dataset_name)
        self.data_args = data_args
        # labels
        # self.label_list = raw_datasets["train"].features["y"].names
        # self.num_labels = len(self.label_list)
        self.num_labels = 1



        raw_datasets = raw_datasets.map(
            preprocess_item,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preprocessing the dataset",
        )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))


        self.data_collator = GraphormerDataCollator()


    def compute_metrics(self, p: EvalPrediction):
        y_true = p.label_ids
        y_pred = p.predictions[:, 1]
        return {"ROC AUC": roc_auc_score(y_true, y_pred)}

class DataSubset(Dataset):
    def __init__(self, root_path, dataset,  dataset_type, path_length=5, split_name=None, split=None, subset_len=None):
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        # Load Data
        df = pd.read_csv(dataset_path)
        if split is not None:
            if dataset in ['molhiv', 'pcba', 'ppa']:
                use_idxs = pd.read_csv(os.path.join(root_path, f"{dataset}/splits/{split}.csv")).values[:,0] - 1
            else:
                use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(self.df))
        if subset_len:
            use_idxs = use_idxs[:subset_len]

        self.df = df.iloc[use_idxs]
        self.smiles = self.df['smiles'].tolist()
        self.use_idxs = use_idxs
        # Dataset Setting
        self.task_names = self.df.columns.drop(['smiles']).tolist()
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        self.dataset_type = dataset_type
        if 'classification' in dataset_type:
            self._task_pos_weights = self.task_pos_weights()
        elif 'regression' in dataset_type:
            self.set_mean_and_std()

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = pmap(smiles2graph,
                          self.smiles)
            avg_size = np.mean([i["num_nodes"] for i in self.graphs])
            print("Nodes per Graph: {}".format(avg_size))
            # self.graphs = []
            # for graph in self.use_idxs:
            #     self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # if isinstance(idx, int):
        # item = smiles2graph(self.smiles[idx])
        item = self.graphs[idx]
        item['y'] = self.labels[idx]
        if 'regression' not in self.dataset_type:
            item['y'] = item['y']
        # item.idx = idx
        return preprocess_item(item)
        # else:
        #     raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

class GraphDataset:
    def __init__(self, data_args, training_args) -> None:
        self.dataset_type = data_args.dataset_type
        self.n_tasks = data_args.n_tasks

        if training_args.do_train:
            self.train_dataset = DataSubset(root_path=data_args.data_path, dataset=data_args.dataset_name,
                                            dataset_type=data_args.dataset_type,
                                            split_name=f'{data_args.split}', split='train',
                                            subset_len=data_args.max_train_samples)
            # if data_args.max_train_samples is not None:
            #     self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            self.eval_dataset = DataSubset(root_path=data_args.data_path, dataset=data_args.dataset_name,
                                           dataset_type=data_args.dataset_type,
                                           split_name=f'{data_args.split}', split='val',
                                           subset_len=data_args.max_eval_samples)
            # if data_args.max_eval_samples is not None:
            #     self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = DataSubset(root_path=data_args.data_path, dataset=data_args.dataset_name,
                                              dataset_type=data_args.dataset_type,
                                              split_name=f'{data_args.split}', split='test',
                                              subset_len=data_args.max_predict_samples)
            # if data_args.max_predict_samples is not None:
            #     self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))


        self.data_collator = GraphormerDataCollator()
        self.metric = data_args.metric
        if data_args.metric is not None:
            self.evaluator = Evaluator(data_args.dataset_name, data_args.metric, self.n_tasks)
        else:
            if "classification" in self.dataset_type:
                self.metric = "AUROC"
                self.evaluator = Evaluator(data_args.dataset_name, "auroc", self.n_tasks)
            elif self.dataset_type == 'multitask':
                self.metric = "AP"
                self.evaluator = Evaluator(data_args.dataset_name, "ap", self.n_tasks)
            else:
                self.metric = "RMSE"
                self.evaluator = Evaluator(data_args.dataset_name, "rmse", self.n_tasks,
                                           mean=self.train_dataset.mean.numpy(),
                                           std=self.train_dataset.std.numpy())

    def compute_metrics(self, p: EvalPrediction):
        y_true = p.label_ids
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(y_true.shape[0], 1)
        y_pred = p.predictions[0]
        return {self.metric: self.evaluator.eval(y_true, y_pred)}

