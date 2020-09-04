import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

GPU_NUM = '2' # np.random.choice(['1', '3'])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
import errno

RANDOM_SEED = 0

import numpy as np
np.random.seed(RANDOM_SEED)

import torch
torch.manual_seed(RANDOM_SEED+1)

from torchvision.utils import save_image

from datetime import datetime

from env import SelfTeachingBaseEnv
from utils import Logger
from sac_multi import SAC_Trainer, ReplayBuffer

if __name__ == '__main__':
    
    num_steps = 300
    dataset = "MNIST"
    pretrained_path = "/opt/workspace/host_storage_hdd/results/2020-06-04_09-07-33_mnist"
    results_path = "/opt/workspace/host_storage_hdd/results"
    save_path = str(datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
    save_path += "_curriculum"
    save_path = os.path.join(results_path, save_path)
    
    logger = Logger()
    logger.set_path(save_path=save_path)
    
    env_kwargs = {"logger": logger,
                  "config_path": './config',
                  "dataset": dataset, 
                  "override_hyperparams": {
                      "random_seed": RANDOM_SEED
                  }
                  }
    
    env = SelfTeachingBaseEnv(**env_kwargs)
    
    replay_buffer = ReplayBuffer(200000)
    
    sac_trainer = SAC_Trainer(replay_buffer, env.observation_space.shape[0], env.action_space.shape[0], logger, hidden_layer_sizes=[128, 128])
    sac_trainer.load_model(os.path.join(pretrained_path, 'best_by_test_sac_self_teaching'), evaluation=True)
    sac_trainer.to_cuda()
    
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action = sac_trainer.policy_net.get_action(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        
        logger.print(step, reward.cpu().detach().numpy())
        
        # create dirs and save images
        os.makedirs(os.path.join(save_path, "epoch_" + str(step)), exist_ok=True)
        
        X = env.X_unlabel
        y = env.y_unlabel
        y_estimates = env.y_unlabel_estimates
        
        action = ((action + 1) / 2).view(-1)
        range_scale = 0.5 - torch.abs(0.5 - action[0])
        action[1] = action[1] * range_scale
        tau1, tau2 = [action[0] - action[1], action[0] + action[1]]
        
        y_unlabel_estimates_max, y_unlabel_estimates_argmax = torch.max(y_estimates, axis=1)
        chosen_indices = (y_unlabel_estimates_max >= tau1) & (y_unlabel_estimates_max <= tau2)
        
        y_chosen = y[chosen_indices]
        y_est_chosen = y_unlabel_estimates_argmax[chosen_indices]
        X_chosen = X[chosen_indices]
        
        correct = y_chosen == y_est_chosen
        
        X_correct = X_chosen[correct]
        X_incorrect = X_chosen[~correct]
        
        # correct
        os.makedirs(os.path.join(save_path, "epoch_" + str(step), "correct"), exist_ok=True)
        for i, (image, _y_true, _y_est) in enumerate(zip(X_correct, y_chosen[correct], y_est_chosen[correct])):
            if i >= 4000:
                break
            filename = "%d_(%d)_(%d).png" % (i, _y_true, _y_est)
            save_image(image, os.path.join(save_path, "epoch_" + str(step), "correct", filename), format="png")
        
        # incorrect
        os.makedirs(os.path.join(save_path, "epoch_" + str(step), "incorrect"), exist_ok=True)
        for i, (image, _y_true, _y_est) in enumerate(zip(X_incorrect, y_chosen[~correct], y_est_chosen[~correct])):
            if i >= 4000:
                break
            filename = "%d_(%d)_(%d).png" % (i, _y_true, _y_est)
            save_image(image, os.path.join(save_path, "epoch_" + str(step), "incorrect", filename), format="png")
        
        step += 1
        