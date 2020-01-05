import warnings
warnings.filterwarnings('ignore')

gpu_num = '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from datetime import datetime
import gym
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
from matplotlib import pyplot as plt

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.sac import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines import PPO2, SAC

from env import SelfTeachingEnv


best_train_mean_episode_rewards = 0
best_test_mean_episode_rewards = 0

def learn_callback(_locals, _globals):
    global best_train_mean_episode_rewards
    global best_test_mean_episode_rewards
    
    reward_lookback = 20
    test_interval = 100
    
    if _locals['episode_rewards'][-1] == 0:
        num_episodes = len(_locals['episode_rewards'])
        mean_episode_rewards = np.mean(_locals['episode_rewards'][-(reward_lookback + 1):-1])
            
        if mean_episode_rewards > best_train_mean_episode_rewards:
            best_mean_episode_rewards = mean_episode_rewards
            _locals['self'].save(save_path + 'best_by_train_sac_self_teaching')
            
        if num_episodes % test_interval == 0:
            accuracies, min_steps, mean_actions, std_actions = test(_locals['self'], _locals['self'].env, n_episodes=5)
            
            is_new_best = False
            if accuracies[-1] > best_test_mean_episode_rewards:
                best_test_mean_episode_rewards = accuracies[-1]
                _locals['self'].save(save_path + 'best_by_test_sac_self_teaching')
                is_new_best = True
            
            plot_actions(mean_actions, std_actions, str(num_episodes) + "_test" + ("_new_best" if is_new_best else ""), "C5")
            plot([accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.8, 1.0), filename=str(num_episodes) + "_test" + ("_new_best" if is_new_best else "") + "_accuracy")
        
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
                
        print("CUMULATIVE REWARD:", rewards_sum, "- NUM STEPS:", num_steps, "- VAL ACCURACY:", info['val_acc'])
        rewards.append(rewards_sum)
        steps.append(num_steps)
        accuracies.append(info['val_acc'])
        
    print("MEAN REWARD:", np.mean(rewards), "- MEAN STEPS:", np.mean(num_steps), "- MEAN ACCURACY:", np.mean(accuracies))
    
    actions = np.array(actions)
    actions_mean = np.mean(actions, axis=0)
    
    env.reset()
    
    return np.mean(return_accuracies, axis=0), np.min(num_steps), np.mean(actions, axis=0), np.std(actions, axis=0)


def plot(arr, labels, y_lim=(0.0, 1.0), filename='basic_RL_results'):
    for data, label in zip(arr, labels):
        plt.plot(np.arange(len(data)), data, label=label)

    plt.legend()
    plt.ylim(y_lim)
    plt.savefig(save_path + filename + '.svg')
                

def plot_actions(mean_actions, std_actions, label, color, save=True):
    
    for i in range(mean_actions.shape[1]):
        plt.plot(np.arange(len(mean_actions[:, i])), mean_actions[:, i], color=color)
        plt.fill_between(np.arange(len(mean_actions[:, i])), mean_actions[:, i] - std_actions[:, i], mean_actions[:, i] + std_actions[:, i], color=color, alpha=0.4)
        
    # plt.legend()
    # plt.ylim((0, 1))
    if save:
        plt.savefig(save_path + label.replace(" ", "_") + "_actions" + ".svg")


if __name__ == '__main__':
    folder_name = str(datetime.now()).replace(" ", "_")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
    os.makedirs(save_path)

    min_steps_array = []
    
    env = make_vec_env(SelfTeachingEnv, n_envs=1, env_kwargs={'EPOCHS_PER_STEP': 2, 'ID': gpu_num, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0}) # , "HISTORY_LEN": 10, "HISTORY_MEAN": True})
    model = SAC(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [256, 512, 256, 128]})
    # model = SAC.load(save_path + '/best_sac_self_teaching_decay_0.05.zip')
    # model.set_env(env)

    """
        # generate_expert_traj(expert_behaviour, 'expert_self_teaching', env, n_episodes=50)
        dataset = ExpertDataset(expert_path=save_path + '/expert_self_teaching.npz', traj_limitation=-1, batch_size=64)

        model.pretrain(dataset, n_epochs=200)
        
        pretrain_accuracies, min_steps, mean_actions, std_actions = test(model, env)
        min_steps_array.append(min_steps)
        plot_actions(mean_actions, std_actions, "pretrained", "C0")
        
        model = SAC.load(save_path + '/best_sac_self_teaching_decay_0.05.zip')    
        basic_trained_accuracies, min_steps, mean_actions, std_actions = test(model, env)
        plot_actions(mean_actions, std_actions, "best", "C1")
        exit()
        min_steps_array.append(min_steps)
        
        model = SAC.load(save_path + '/best_sac_self_teaching_1.zip')
        ep_limit_accuracies, min_steps = test(model, env)
        min_steps_array.append(min_steps)
        
        labeled_only_accuracies, min_steps = test(model, env, override_action=True)
        min_steps_array.append(min_steps)
        
        min_steps = np.min(min_steps_array)
        arr_to_plot = []
        arr_to_plot.append(pretrain_accuracies[:min_steps])
        arr_to_plot.append(basic_trained_accuracies[:min_steps])
        # arr_to_plot.append(ep_limit_accuracies[:min_steps])
        arr_to_plot.append(labeled_only_accuracies[:min_steps])
        
        plot(arr_to_plot, labels=["pretrained", "RL trained - best", "label only baseline"], y_lim=(0.8, 1.0))
        exit()
    """
    
    model.learn(total_timesteps=500000, log_interval=10, callback=learn_callback)
    model.save(save_path + 'final_sac_self_teaching')

    accuracies, min_steps, mean_actions, std_actions = test(model, env)
    plot_actions(mean_actions, std_actions, "final", "C5")
    plot([accuracies], labels=["final"], y_lim=(0.8, 1.0))
