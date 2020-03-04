# This implements transfer learning with differing number of clusters in data. 
# Namely, we transfer the model from 1 cluster to N clusters.
import warnings
warnings.filterwarnings('ignore')

gpu_num = '1,2,3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

from datetime import datetime
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

from multiprocessing import Process
from multiprocessing.managers import BaseManager

import sac_multi
from sac_multi import SAC_Trainer, ReplayBuffer, share_parameters, worker
from env import SelfTeachingEnvV1 as SelfTeachingEnv
from utils import save_self, test_pipeline, learn_callback


def transfer_weights(old_trainer, new_trainer, noise=1.0):
    # this only works with SAC_Trainer
    assert isinstance(old_trainer, SAC_Trainer)
    assert isinstance(new_trainer, SAC_Trainer)
    assert hasattr(old_trainer, "policy_net") and isinstance(old_trainer.policy_net, sac_multi.PolicyNetwork)
    assert hasattr(old_trainer, "value_net") and isinstance(old_trainer.value_net, sac_multi.ValueNetwork)
    assert hasattr(old_trainer, "target_value_net") and isinstance(old_trainer.target_value_net, sac_multi.ValueNetwork)
    assert hasattr(old_trainer, "q_net_1") and isinstance(old_trainer.q_net_1, sac_multi.SoftQNetwork)
    assert hasattr(old_trainer, "q_net_2") and isinstance(old_trainer.q_net_2, sac_multi.SoftQNetwork)
    assert hasattr(old_trainer, "log_alpha") and isinstance(old_trainer.log_alpha, sac_multi.Alpha)
    assert hasattr(new_trainer, "policy_net") and isinstance(new_trainer.policy_net, sac_multi.PolicyNetwork)
    assert hasattr(new_trainer, "value_net") and isinstance(new_trainer.value_net, sac_multi.ValueNetwork)
    assert hasattr(new_trainer, "target_value_net") and isinstance(new_trainer.target_value_net, sac_multi.ValueNetwork)
    assert hasattr(new_trainer, "q_net_1") and isinstance(new_trainer.q_net_1, sac_multi.SoftQNetwork)
    assert hasattr(new_trainer, "q_net_2") and isinstance(new_trainer.q_net_2, sac_multi.SoftQNetwork)
    assert hasattr(new_trainer, "log_alpha") and isinstance(new_trainer.log_alpha, sac_multi.Alpha)
    
    new_trainer.policy_net.copy_weights(old_trainer.policy_net, noise_weight=noise)
    
    new_trainer.value_net.copy_weights(old_trainer.value_net)
    new_trainer.target_value_net.copy_weights(old_trainer.target_value_net)
    
    new_trainer.q_net_1.copy_weights(old_trainer.q_net_1, noise_weight=noise)
    new_trainer.q_net_2.copy_weights(old_trainer.q_net_2, noise_weight=noise)
    
    
def freeze_parameters(trainer, freeze_mask):
    trainer.policy_net.freeze(freeze_mask)
    # trainer.value_net.freeze(freeze_mask)
    # trainer.target_value_net.freeze(freeze_mask)
    # trainer.q_net_1.freeze(freeze_mask)
    # trainer.q_net_2.freeze(freeze_mask)    


if __name__ == '__main__':
    os.system('clear')

    env_kwargs_old = {'EPOCHS_PER_STEP': 2, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0, 'N_CLUSTERS': 1}
    env_kwargs_new = {'EPOCHS_PER_STEP': 2, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0, 'N_CLUSTERS': 10}
    worker_offset = 0
    num_workers = 9
    initial_lr = 1e-5
    final_lr = 1e-5
    num_steps = 1500000
    learning_starts = 30000
    n_warmup = 0
    batch_size = 8
    rl_hidden_layer_sizes = [128, 128]
    buffer_size = 500000
    
    # test_pipeline('/opt/workspace/host_storage_hdd/results/2020-02-16_15-14-10.497811/')
    # exit()
    
    folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
    os.makedirs(save_path, exist_ok=True)
    save_self(save_path)
    
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(buffer_size)

    env_old = SelfTeachingEnv(**env_kwargs_old)
    action_dim_old = env_old.action_space.shape[0]
    state_dim_old  = env_old.observation_space.shape[0]
    env_old.close()
    del env_old
    
    env_new = SelfTeachingEnv(**env_kwargs_new)
    action_dim_new = env_new.action_space.shape[0]
    state_dim_new = env_new.observation_space.shape[0]
    env_new.close()
    del env_new

    sac_trainer_old = SAC_Trainer(replay_buffer, state_dim_old, action_dim_old, hidden_layer_sizes=rl_hidden_layer_sizes, q_lr=initial_lr, pi_lr=initial_lr, alpha_lr=initial_lr, v_lr=initial_lr)
    sac_trainer_old.load_model('/opt/workspace/host_storage_hdd/results/2020-02-17_09-32-42.654662_128x128_best/best_by_test_sac_self_teaching')
    
    sac_trainer_new = SAC_Trainer(replay_buffer, state_dim_new, action_dim_new, hidden_layer_sizes=rl_hidden_layer_sizes, q_lr=initial_lr, pi_lr=initial_lr, alpha_lr=initial_lr, v_lr=initial_lr)
    
    transfer_weights(sac_trainer_old, sac_trainer_new, noise=0.0)
    freeze_parameters(sac_trainer_new, [True] * len(rl_hidden_layer_sizes) + [False])
    
    del sac_trainer_old
    
    sac_trainer_new.q_net_2.share_memory()
    sac_trainer_new.q_net_1.share_memory()
    sac_trainer_new.policy_net.share_memory()
    sac_trainer_new.value_net.share_memory()
    sac_trainer_new.target_value_net.share_memory()
    sac_trainer_new.log_alpha.share_memory()
    share_parameters(sac_trainer_new.q_optimizer_1)
    share_parameters(sac_trainer_new.q_optimizer_2)
    share_parameters(sac_trainer_new.policy_optimizer)
    share_parameters(sac_trainer_new.alpha_optimizer)
    share_parameters(sac_trainer_new.v_optimizer)
    
    processes = []
    for i in range(worker_offset, num_workers + worker_offset):
        process = Process(target=worker,
                          kwargs={'worker_id': i,
                                  'sac_trainer': sac_trainer_new,
                                  'env_fn': SelfTeachingEnv,
                                  'env_kwargs': env_kwargs_new,
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
        
    test_pipeline(SelfTeachingEnv(**env_kwargs_new), sac_trainer_new, model_path=save_path)
    
    
    
    