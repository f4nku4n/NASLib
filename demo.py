import json
import logging
import os

# import the Trainer used to run the optimizer on a given search space
from naslib.defaults.trainer import Trainer
# import the optimizers
from naslib.optimizers import (
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    # BasePredictor,
    Npenas
)
# import the search spaces
from naslib.search_spaces import (
    # NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBenchASRSearchSpace,
    # NasBench301SearchSpace,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib import utils
from naslib.utils import get_dataset_api
from naslib.utils.log import setup_logger

from fvcore.common.config import CfgNode # Required to read the config
###### End of imports ######

# Set up the seeds
utils.set_seed(9002)

# Instantiate the search space and get its benchmark API
search_space = NasBench201SearchSpace()

# dataset_api = get_api_data('nasbench201', 'cifar10')

import pickle
with open('/content/drive/MyDrive/QuanPM/Benchmark_Data/performance_predictors_datasets/nb201_cifar10_full_training.pickle', 'rb') as f:
    data = pickle.load(f)
dataset_api = {"nb201_data": data}

# Instantitate the optimizer and adapt the search space to it

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
# logger.info(f'Configuration is \n{config}')

optimizer = RandomSearch(config)
# optimizer = Npenas(config)
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

# Create a Trainer
trainer = Trainer(optimizer, config)

# Perform the search
trainer.search(resume_from="", report_incumbent=False)

# Get the results of the search
search_trajectory = trainer.search_trajectory
print('Train accuracies:', search_trajectory.train_acc)
print('Validation accuracies:', search_trajectory.valid_acc)

# Get the validation performance of the best model found in the search phase
best_model_val_acc = trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)
best_model_val_acc