import warnings
warnings.filterwarnings('ignore')

GPU_NUM = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

import argparse

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from env import SelfTeachingBaseEnv
from utils import Logger, test_pipeline

if __name__ == "__main__":
    os.system('clear')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config", action="store", help="Folder containing config file.")
    parser.add_argument("--random-seed", type=int, default=np.random.randint(100000), action="store", help="Random seed used for experiment initalization. Recommended when continuing training.")
    parser.add_argument("--results-folder", type=str, default="/opt/workspace/host_storage_hdd/results", action="store", help="Folder used as base when creating the results directory.")
    parser.add_argument("--path-postfix", type=str, default="model_testing", action="store", help="Path postfix that is added to the logging folder.")
    
    args = parser.parse_args()
    args.dataset = "ag_news"
    args.from_dataset = ""
    
    logger = Logger()
    logger.create_logdirs(args)
    
    env = SelfTeachingBaseEnv(config_path=args.config_path, dataset=args.dataset, override_hyperparams={"random_seed": args.random_seed})
    
    test_pipeline(env=env, trainer=None, logger=logger, all_samples=False, manual_thresholds=False, labeled_samples=False, trained_model=False, conf_matrix=False, n_test_runs=2)