import warnings
warnings.filterwarnings('ignore')

gpu_num = '2'
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ddpg import TD3Trainer
from ray.tune.logger import pretty_print

from env import SelfTeachingEnvV0


if __name__ == "__main__":
    ray.init()
    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = 3
    config["num_workers"] = 8
    config['log_level'] = 'ERROR'
    config['log_sys_usage'] = False
    
    # config['Q_model']['hidden_activation'] = 'relu'
    config['Q_model']['hidden_layer_sizes'] = (512, 1024, 512, 256)
    # config['policy_model']['hidden_activation'] = 'relu'
    config['policy_model']['hidden_layer_sizes'] = (512, 1024, 512, 256)
    
    # config['learning_starts'] = 10000
    
    # config['env_config'] = {'ID': gpu_num, 'SIGNIFICANCE_DECAY': 0.0, 'N_TIMESTEPS': 150, 'EPOCHS_PER_STEP': 2}
    # config['twin_q'] = True
    # config['actor_hiddens'] = [512, 1024, 512, 256]
    # config['critic_hiddens'] = [512, 1024, 512, 256]
    # config['exploration_noise_type'] = 'gaussian'
    # config['buffer_size'] = 100000
    # config['learning_starts'] = 10000
    
    # trainer = TD3Trainer(config=config, env=SelfTeachingEnvV0)
    trainer = SACTrainer(config=config, env='MountainCarContinuous-v0')

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
            
    exit()
    ray.init(plasma_directory='/opt/workspace/host_storage_hdd/plasma',
             temp_dir='/opt/workspace/host_storage_hdd/ray')
    
    config = DEFAULT_CONFIG.copy()
    # config['env'] = SelfTeachingEnvV0
    
    config['num_workers'] = 8
    config['num_gpus'] = 3
    
# sac specific args:
"""
'Q_model': {
    'hidden_activation': 'relu',
    'hidden_layer_sizes': (512, 1024, 512, 256)
},
'policy_model': {
    'hidden_activation': 'relu',
    'hidden_layer_sizes': (512, 1024, 512, 256)
},
'timesteps_per_iteration': 100,
'learning_starts': 10000
"""    
"""
tune.run(TD3Trainer, config={'env': SelfTeachingEnvV0, 
                                'env_config': {'ID': gpu_num, 'SIGNIFICANCE_DECAY': 0.0, 'N_TIMESTEPS': 150, 'EPOCHS_PER_STEP': 2},
                                'num_workers': 8,
                                'num_gpus': 2,
                                'log_level': 'ERROR',
                                'evaluation_interval': 50,
                                'evaluation_num_episodes': 5,
                                'log_sys_usage': False,
                                'compress_observations': True,
                                'callbacks': {
                                    # 'on_episode_step': train_callback
                                },
                                
                                # td3 specific args
                                'twin_q': True,
                                'actor_hiddens': [512, 1024, 512, 256],
                                'critic_hiddens': [512, 1024, 512, 256],
                                'exploration_noise_type': 'gaussian',
                                
                                'buffer_size': 100000,
                                'learning_starts': 10000
                                })
"""
