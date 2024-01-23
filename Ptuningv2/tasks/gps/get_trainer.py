import logging
from model.configuration_gps import GraphGPSConfig

from model.utils import get_model, TaskType
from .dataset import MoleculeNetDataset
from training.trainer_base import PYGBaseTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    dataset = MoleculeNetDataset(data_args, training_args)

    if dataset.dataset_type == 'binary classification':
        config = GraphGPSConfig(n_tasks=dataset.n_tasks, task_type=dataset.dataset_type)
    elif dataset.dataset_type == 'multi-task classification':
        config = GraphGPSConfig(n_tasks=dataset.n_tasks, task_type=dataset.dataset_type,
                             pre_seq_len=model_args.pre_seq_len, prefix_projection=None)
    else:
        config = GraphGPSConfig(n_tasks=dataset.n_tasks, task_type=dataset.dataset_type)

    model = get_model(model_args, TaskType.GRAPH, config)

    # Initialize our Trainer
    trainer = PYGBaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        data_collator=dataset.data_collator,
        test_key=dataset.metric
    )

    return trainer, dataset