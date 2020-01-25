import warnings
warnings.filterwarnings('ignore')

gpu_num = '1,3,2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from datetime import datetime
import shutil
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from matplotlib import pyplot as plt

import torch.multiprocessing as mp

from multiprocessing import Process
from multiprocessing.managers import BaseManager

# from mpi4py import MPI

"""
# from stable_baselines.sac import MlpPolicy
# from stable_baselines.deepq import MlpPolicy
from stable_baselines.ddpg import MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2, SAC, DQN, DDPG, ACKTR, logger
from stable_baselines.common.vec_env import SubprocVecEnv
"""

from env import SelfTeachingEnvV0 as SelfTeachingEnv
from sac_multi import SAC_Trainer, ReplayBuffer, share_parameters, worker

best_train_mean_episode_rewards = 0
best_test_mean_episode_rewards = 0

def learn_callback(_locals, _globals):
    global best_train_mean_episode_rewards
    global best_test_mean_episode_rewards
    
    if _locals['episode_reward'] == 0 and _locals['worker_id'] == 0:
        num_episodes = _locals['n_episodes'] + 1
        if _locals['step'] < _locals['learning_starts']:
            return True
 
        reward_lookback = 20
        # test_interval = int(5 + (1 - (['step'] - _locals['self'].learning_starts) / (_locals['total_timesteps'] - _locals['self'].learning_starts)) * 65)
        test_interval = 40
        test_episodes = 5
        
        mean_episode_rewards = np.mean(_locals['rewards'][-reward_lookback:])
            
        if mean_episode_rewards > best_train_mean_episode_rewards:
            best_mean_episode_rewards = mean_episode_rewards
            _locals['sac_trainer'].save_model(_locals['log_path'] + 'best_by_train_sac_self_teaching')
        
        if num_episodes % test_interval == 0:
            mean_accuracies, std_accuracies, mean_actions, std_actions = test(_locals['sac_trainer'], _locals['env'], n_episodes=test_episodes, scale=True)

            if mean_accuracies[-1] > best_test_mean_episode_rewards:
                best_test_mean_episode_rewards = mean_accuracies[-1]
                _locals['sac_trainer'].save_model(_locals['log_path'] + 'best_by_test_sac_self_teaching')

                plot_actions(mean_actions, std_actions, label=str(num_episodes) + "_test", color="C5", filepath=_locals['log_path'])
                plot([mean_accuracies], [std_accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.8, 1.0), filename=str(num_episodes) + "_test" + "_accuracy_%.4f" % (mean_accuracies[-1]), filepath=_locals['log_path'])
            
            if 'writer' in _locals:
                writer = _locals['writer']
                writer.add_scalar('Actions/meanTestActions', np.mean(mean_actions), _locals['step'])
                writer.add_scalar('Accuracies/testAccuracies', mean_accuracies[-1], _locals['step'])
    
    return True


def expert_behaviour(obs):
    tau1 = 0.99
    tau2 = 0.992
    
    output_action = np.ones((obs.shape[0], 2))
    output_action[:, 0] *= tau1
    output_action[:, 1] *= tau2
    
    output_action += np.random.normal(scale=0.005, size=output_action.shape)
    return output_action


def test(model, env, n_episodes=10, override_action=False, scale=False):
    rewards = []
    steps = []
    accuracies = []
    actions = []
    
    return_accuracies = np.zeros((n_episodes, 150))
    for i in range(n_episodes):        
        obs = env.reset()
        done = False
        rewards_sum = 0
        num_steps = 0
        actions.append([])
        while not done:
            env.render()
                        
            action = model.policy_net.get_action(obs, deterministic=True)
            if override_action:
                if isinstance(override_action, list):
                    action = np.array([override_action])
                else:
                    action = np.zeros(2).reshape((1, -1))
                    
            obs, reward, done, info = env.step(action)
            reward = reward if isinstance(reward, float) else reward[0]
            info = info if isinstance(info, dict) else info[0]
            action = action.reshape(-1)
            
            rewards_sum += reward
            num_steps += 1
            
            if info['timestep'] < 150:
                return_accuracies[i, info['timestep']] = info['val_acc']
            
            actions[-1].append(action.tolist())
                
        print(i, ": CUMULATIVE REWARD:", rewards_sum, "- NUM STEPS:", num_steps, "- VAL ACCURACY:", info['val_acc'])
        rewards.append(rewards_sum)
        steps.append(num_steps)
        accuracies.append(info['val_acc'])
        
    print("MEAN REWARD:", np.mean(rewards), "- MEAN STEPS:", np.mean(num_steps), "- MEAN ACCURACY:", np.mean(accuracies))
    
    actions = np.array(actions)
    if scale:
        actions = (actions + 1) / 2
    
    env.reset()
    
    return np.mean(return_accuracies, axis=0), np.std(return_accuracies, axis=0), np.mean(actions, axis=0), np.std(actions, axis=0)


def plot(mean_arr, std_arr, labels, y_lim=(0.0, 1.0), filename='RL_results', filepath=None):
    plt.clf()
    for mean_data, std_data, label in zip(mean_arr, std_arr, labels):
        plt.plot(np.arange(len(mean_data)), mean_data, label=label)
        plt.fill_between(np.arange(len(std_data)), mean_data-std_data, mean_data+std_data, alpha=0.4)

    plt.legend()
    plt.ylim(y_lim)
    if filepath is not None:
        plt.savefig(filepath + filename + '.svg')
    else:
        plt.savefig(save_path + filename + '.svg')
                

def plot_actions(mean_actions, std_actions, label, color, save=True, filepath=None):
    plt.clf()
    for i in range(mean_actions.shape[1]):
        plt.plot(np.arange(len(mean_actions[:, i])), mean_actions[:, i], color=color)
        plt.fill_between(np.arange(len(mean_actions[:, i])), mean_actions[:, i] - std_actions[:, i], mean_actions[:, i] + std_actions[:, i], color=color, alpha=0.4)
        
    # plt.legend()
    # plt.ylim((0, 1))
    if save:
        if filepath is not None:
            plt.savefig(filepath + label.replace(" ", "_") + "_actions" + ".svg")
        else:
            plt.savefig(save_path + label.replace(" ", "_") + "_actions" + ".svg")


def test_pipeline(model_path):
    env = make_vec_env(SelfTeachingEnv, n_envs=1, env_kwargs={'EPOCHS_PER_STEP': 2, 'ID': gpu_num, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0})
    mean_accs, std_accs = [], []
    model = SAC.load(model_path)
    
    mean_acc, std_acc, mean_actions, std_actions = test(model, env, override_action=[0.99, 0.992], n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
        
    mean_acc, std_acc, mean_actions, std_actions = test(model, env, n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    mean_acc, std_acc, mean_actions, std_actions = test(model, env, override_action=True, n_episodes=10)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    plot(mean_accs, std_accs, labels=["manually set thresholds", "RL trained", "label only baseline"], y_lim=(0.8, 1.0), filename=model_path.split('/')[-2], filepath='/opt/workspace/host_storage_hdd/results/')


def save_self(filepath):
    filepath = filepath + 'code/'
    os.makedirs(filepath, exist_ok=True)
    filenames = ['main.py', 'model.py', 'env.py', 'sac_multi.py', 'preprocess.py']
    
    for filename in filenames:
        shutil.copyfile(filename, filepath + filename)
    

if __name__ == '__main__':
    # test_pipeline('/opt/workspace/host_storage_hdd/results/2020-01-08_17:30:00.436350/best_by_train_sac_self_teaching.zip')
    # exit()
    
    folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
    os.makedirs(save_path, exist_ok=True)
    save_self(save_path)
    
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(100000)

    # env_kwargs = {'EPOCHS_PER_STEP': 2, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0}
    env_kwargs = {'id': 'MountainCarContinuous-v0'}
    # env = SelfTeachingEnv(**env_kwargs)
    import gym
    env = gym.make(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    env.close()

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=[512, 1024, 512, 256])

    sac_trainer.q_net_1.share_memory()
    sac_trainer.q_net_2.share_memory()
    sac_trainer.policy_net.share_memory()
    sac_trainer.value_net.share_memory()
    sac_trainer.target_value_net.share_memory()
    share_parameters(sac_trainer.q_optimizer_1)
    share_parameters(sac_trainer.q_optimizer_2)
    share_parameters(sac_trainer.policy_optimizer)
    share_parameters(sac_trainer.alpha_optimizer)
    share_parameters(sac_trainer.v_optimizer)

    worker_offset = 0
    num_workers = 1
    processes = []
    for i in range(worker_offset, num_workers + worker_offset):
        process = Process(target=worker,
                          kwargs={'worker_id': i,
                                  'sac_trainer': sac_trainer,
                                  'env_fn': gym.make, # SelfTeachingEnv, # gym.make,
                                  'env_kwargs': env_kwargs,
                                  'replay_buffer': replay_buffer,
                                  'num_steps': 300000 // num_workers,
                                  'learning_starts': 100 // num_workers,
                                  'callback': learn_callback,
                                  'log_path': save_path}
                          )
        process.daemon = True
        processes.append(process)
        
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    
"""
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:    
        folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-")
        save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
        os.makedirs(save_path, exist_ok=True)
        save_self(save_path)
        logger.configure()
        
        size = MPI.COMM_WORLD.Get_size()
        for i in range(1, size):
            MPI.COMM_WORLD.send(save_path, dest=i)
    else:
        logger.configure(format_strs=[])
        save_path=MPI.COMM_WORLD.recv(source=0)
    
    env = make_vec_env(SelfTeachingEnv, n_envs=1, env_kwargs={'EPOCHS_PER_STEP': 2, 'ID': gpu_num[-1], 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0})
    
    model = ACKTR(MlpPolicy, env,
                tensorboard_log=save_path,
                verbose=1,
                policy_kwargs={'layers': [512, 1024, 512, 256]}
                )
                
    model = DDPG(MlpPolicy, 
                 env,
                 buffer_size=100000,
                 observation_range=(0.0, 10.0),
                 tensorboard_log=save_path,
                 verbose=1,
                 policy_kwargs={'layers': [512, 1024, 512, 256]})
    
    model = SAC(MlpPolicy, 
                env, 
                learning_starts=5000, 
                buffer_size=100000,
                learning_rate=lambda x : 0.00001 + x * 0.00034,
                tau=0.003,
                tensorboard_log=save_path,
                verbose=1,
                policy_kwargs={'layers': [512, 1024, 512, 256]}
                )
    
    model.learn(total_timesteps=400000, log_interval=10, callback=learn_callback)
    if rank == 0:
        model.save(save_path + 'final_sac_self_teaching')

        mean_accuracies, std_accuracies, mean_actions, std_actions = test(model, env)
        plot_actions(mean_actions, std_actions, "final", "C5")
        plot([mean_accuracies], [std_accuracies], labels=["final"], y_lim=(0.8, 1.0), filename='final_RL_results')
"""
