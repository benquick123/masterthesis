import warnings
warnings.filterwarnings('ignore')

GPU_NUM = '0'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

import argparse
import pickle

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from utils import test, plot, plot_actions, Logger
from env import SelfTeachingBaseEnv


def run_tests(env, logger, args):
    best_accuracy = args.min_accuracy
    best_thresholds = (None, None)
    
    for iteration in range(args.start_from, args.num_iterations):
        logger.print("Iteration:", iteration)
        action = np.random.uniform(low=-1.0, high=1.0, size=(2,)).tolist()
        logger.print("Testing thresholds:", action)
        mean_accuracies, _, _, _, _, _ = test(None, env, logger=logger, override_action=action, n_episodes=5)
        
        if mean_accuracies[-1] > best_accuracy:
            best_accuracy = mean_accuracies[-1]
            best_thresholds = tuple(action)
            logger.print("NEW BEST:", best_accuracy, "", str(best_thresholds))
        
        
    logger.print("Performing final test.")
    mean_accs, std_accs, mean_actions, std_actions, mean_samples, std_samples = test(None, env, logger=logger, override_action=list(best_thresholds), n_episodes=10)
    
    if logger.save_path:
        labels = ['best random thresholds']
        plot([mean_accs], [std_accs], labels=labels, y_lim=(0.0, 1.0), filename="test_curves", filepath=logger.save_path)
        plot([mean_samples], [std_samples], labels=["num selected samples"], y_lim=(0, 1), filename='test_samples', filepath=logger.save_path)
        plot_actions(mean_actions, std_actions, label="test", color="C5", filepath=logger.save_path)
        
        pickle.dump({"mean_accs": mean_accs, "std_accs": std_accs, 
                     "labels": labels, 
                     "mean_actions": mean_actions, "std_actions": std_actions, 
                     "mean_samples": mean_samples, "std_samples": std_samples}, 
                    open(os.path.join(logger.save_path, "test_results.pkl"), "wb"))


if __name__ == '__main__':
    os.system('clear')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action="store", required=True, help="Dataset to use.")
    parser.add_argument("--num-iterations", type=int, default=386, help="How many random threshold runs to try out.")
    parser.add_argument("--path-postfix", type=str, default="baseline", action="store", help="Path postfix that is added to the logging folder.")
    parser.add_argument("--config-path", type=str, default="./config", action="store", help="Folder containing config file.")
    parser.add_argument("--results-folder", type=str, default="/opt/workspace/host_storage_hdd/results", action="store", help="Folder used as base when creating the results directory.")
    parser.add_argument("--random-seed", type=int, default=np.random.randint(100000), action="store", help="Random seed used for experiment initalization. Recommended when continuing training.")
    parser.add_argument("--start-from", type=int, default=0, action="store", help="Iteration from which to start testing.")
    parser.add_argument("--min-accuracy", type=float, default=0.0, action="store", help="Minimum accuracy that triggers 'NEW BEST'.")
    
    args = parser.parse_args()
    args.from_dataset = ""
    
    logger = Logger()
    logger.create_logdirs(args)
    
    env = SelfTeachingBaseEnv(logger=logger, config_path=args.config_path, dataset=args.dataset, override_hyperparams={"random_seed": args.random_seed})
    
    run_tests(env, logger, args)