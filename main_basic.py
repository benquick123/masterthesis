import warnings
warnings.filterwarnings('ignore')

gpu_num = '1,2,3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

import argparse
from datetime import datetime
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

import torch.multiprocessing as mp
from torchvision import transforms

from multiprocessing import Process
from multiprocessing.managers import BaseManager

from env import SelfTeachingBaseEnv
from sac_multi import SAC_Trainer, ReplayBuffer, share_parameters, worker
from utils import save_self, learn_callback, test_pipeline


if __name__ == '__main__':
    os.system('clear')
    
    parser = argparse.ArgumentParser(description="Runner for one dataset pipeline.")
    parser.add_argument("--dataset", type=str, action="store", required=True, help="Dataset to use.")
    args = parser.parse_args()

    """
        N_CLUSTERS = 1
        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                            ElasticTransform(5, 35),
                                            transforms.RandomChoice([transforms.RandomAffine(7.5), transforms.RandomAffine(0, shear=7.5)]),
                                            transforms.RandomSizedCrop(28, scale=(0.8, 1.2)),
                                            transforms.ToTensor()])
    """
    env_kwargs = {"dataset": args.dataset, 
                  "override_hyperparams": {
                      "reward_scale": 10.0
                      }
                  }
    
    worker_offset = 0
    num_workers = 1
    initial_lr = 5e-5
    final_lr = 5e-5
    num_steps = 500000
    learning_starts = 10000
    n_warmup = learning_starts
    batch_size = 16
    rl_hidden_layer_sizes = [128, 128]
    buffer_size = 100000
    
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(buffer_size)

    env = SelfTeachingBaseEnv(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    env.close()

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=rl_hidden_layer_sizes, q_lr=initial_lr, pi_lr=initial_lr, alpha_lr=initial_lr, v_lr=initial_lr)
    
    # test_pipeline(env, sac_trainer, '/opt/workspace/host_storage_hdd/results/2020-03-05_10-28-29.855557_fashion_mnist/')
    # exit()
    del env
        
    folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '_' + env_kwargs['dataset'] + '/'
    os.makedirs(save_path, exist_ok=True)
    save_self(save_path)

    sac_trainer.q_net_1.share_memory()
    sac_trainer.q_net_2.share_memory()
    sac_trainer.policy_net.share_memory()
    sac_trainer.value_net.share_memory()
    sac_trainer.target_value_net.share_memory()
    sac_trainer.log_alpha.share_memory()
    share_parameters(sac_trainer.q_optimizer_1)
    share_parameters(sac_trainer.q_optimizer_2)
    share_parameters(sac_trainer.policy_optimizer)
    share_parameters(sac_trainer.alpha_optimizer)
    share_parameters(sac_trainer.v_optimizer)
    
    processes = []
    for i in range(worker_offset, num_workers + worker_offset):
        process = Process(target=worker,
                          kwargs={'worker_id': i,
                                  'sac_trainer': sac_trainer,
                                  'env_fn': SelfTeachingBaseEnv,
                                  'env_kwargs': env_kwargs,
                                  'replay_buffer': replay_buffer,
                                  'num_steps': num_steps // num_workers,
                                  'learning_starts': learning_starts // num_workers,
                                  'n_warmup': n_warmup // num_workers,
                                  # 'linear_lr_scheduler': [initial_lr, final_lr],
                                  'n_updates': num_workers,
                                  'batch_size': batch_size,
                                  'callback': learn_callback,
                                  'log_path': save_path}
                          )
        process.daemon = True
        processes.append(process)
        
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    
    
        
    test_pipeline(SelfTeachingBaseEnv(**env_kwargs), sac_trainer, save_path)

