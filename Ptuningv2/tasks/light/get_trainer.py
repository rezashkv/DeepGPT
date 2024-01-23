import logging
from model.configuration_light import LiGhTConfig

from model.utils import get_model, TaskType
from tasks.light.dataset import MoleculeDataset
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    dataset = MoleculeDataset(data_args, training_args)


    config = LiGhTConfig(n_tasks=dataset.n_tasks, task_type=dataset.dataset_type,
                         pre_seq_len=model_args.pre_seq_len, prefix_projection=None,
                         light_weight_tuning=model_args.lightweight)


    model = get_model(model_args, TaskType.GRAPH, config)


    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        data_collator=dataset.data_collator,
        test_key=dataset.metric
    )

    return trainer, dataset