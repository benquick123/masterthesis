import warnings
warnings.filterwarnings('ignore')

gpu_num = '2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from datetime import datetime
import shutil
import gym
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
from matplotlib import pyplot as plt

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.sac import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2, SAC

from env import SelfTeachingEnvV1 as SelfTeachingEnv

best_train_mean_episode_rewards = 0
best_test_mean_episode_rewards = 0

def learn_callback(_locals, _globals):
    global best_train_mean_episode_rewards
    global best_test_mean_episode_rewards
    
    if _locals['episode_rewards'][-1] == 0:        
        num_episodes = len(_locals['episode_rewards'])
        if _locals['step'] < _locals['self'].learning_starts:
            return True
        
        reward_lookback = 20
        test_interval = int(5 + (1 - (_locals['step'] - _locals['self'].learning_starts) / (_locals['total_timesteps'] - _locals['self'].learning_starts)) * 65)
        test_interval = 1
        test_episodes = 5
        
        mean_episode_rewards = np.mean(_locals['episode_rewards'][-(reward_lookback + 1):-1])
            
        if mean_episode_rewards > best_train_mean_episode_rewards:
            best_mean_episode_rewards = mean_episode_rewards
            _locals['self'].save(save_path + 'best_by_train_sac_self_teaching')
            
        if num_episodes % test_interval == 0:
            mean_accuracies, std_accuracies, mean_actions, std_actions = test(_locals['self'], _locals['self'].env, n_episodes=test_episodes)
            
            is_new_best = False
            if mean_accuracies[-1] > best_test_mean_episode_rewards:
                best_test_mean_episode_rewards = mean_accuracies[-1]
                _locals['self'].save(save_path + 'best_by_test_sac_self_teaching')
                is_new_best = True
            
            # plot_actions(mean_actions, std_actions, str(num_episodes) + "_test" + ("_new_best" if is_new_best else ""), "C5")
            # plot([mean_accuracies], [std_accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.8, 1.0), filename=str(num_episodes) + "_test" + ("_new_best" if is_new_best else "") + "_accuracy")
        
    return True


def expert_behaviour(obs):
    tau1 = 0.99
    tau2 = 0.992
    
    output_action = np.ones((obs.shape[0], 2))
    output_action[:, 0] *= tau1
    output_action[:, 1] *= tau2
    
    output_action += np.random.normal(scale=0.005, size=output_action.shape)
    return output_action


def test(model, env, n_episodes=10, override_action=False):
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
                        
            action, _states = model.predict(obs)
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
    actions_mean = np.mean(actions, axis=0)
    
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
    os.makedirs(filepath)
    filenames = ['main.py', 'model.py', 'env.py', 'preprocess.py']
    
    for filename in filenames:
        shutil.copyfile(filename, filepath + filename)
    

if __name__ == '__main__':
    # test_pipeline('/opt/workspace/host_storage_hdd/results/2020-01-08_17:30:00.436350/best_by_train_sac_self_teaching.zip')
    # exit()
    
    folder_name = str(datetime.now()).replace(" ", "_")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
    os.makedirs(save_path)
    save_self(save_path)
    
    env = make_vec_env(SelfTeachingEnv, 
                       n_envs=1, 
                       env_kwargs={'EPOCHS_PER_STEP': 2, 'ID': gpu_num, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0}) # , "HISTORY_LEN": 10, "HISTORY_MEAN": True})
    
    model = SAC(MlpPolicy, 
                env,
                learning_starts=1,
                learning_rate=lambda x : 0.00001 + x * 0.00034, 
                # tau=0.003, 
                verbose=1, 
                policy_kwargs={'layers': [256, 512, 256, 128]})
    """
    model = SAC.load('/opt/workspace/host_storage_hdd/results/2020-01-07_19:21:48.837678/best_by_test_sac_self_teaching.zip')
    model.set_env(env)
    model.learning_rate = lambda x : 0.00001 + x * 0.00002
    model.learning_starts = 5000
    model.tau = 0.003
    """
    model.learn(total_timesteps=400000, log_interval=10, callback=learn_callback)
    model.save(save_path + 'final_sac_self_teaching')

    mean_accuracies, std_accuracies, mean_actions, std_actions = test(model, env)
    plot_actions(mean_actions, std_actions, "final", "C5")
    plot([mean_accuracies], [std_accuracies], labels=["final"], y_lim=(0.8, 1.0), filename='final_RL_results')
