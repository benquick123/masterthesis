import warnings
warnings.filterwarnings('ignore')

gpu_num = '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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


best_mean_episode_rewards = 0
def learn_callback(_locals, _globals):
    global best_mean_episode_rewards

    if len(_locals['episode_rewards']) < 100:
        mean_episode_rewards = np.mean(_locals['episode_rewards'])
    else:
        mean_episode_rewards = np.mean(_locals['episode_rewards'][-100:])
        
    if mean_episode_rewards > best_mean_episode_rewards:
        best_mean_episode_rewards = mean_episode_rewards
        _locals['self'].save('/opt/workspace/host_storage_hdd/best_sac_self_teaching_' + str(gpu_num))
        
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
    
    return_accuracies = np.zeros((n_episodes, 150))
    for i in range(n_episodes):        
        obs = env.reset()
        done = False
        rewards_sum = 0
        num_steps = 0
        while not done:
            env.render()
                        
            action, _states = model.predict(obs)
            if override_action:
                action = np.array([[0.0, 0.0]])
            obs, reward, done, info = env.step(action)
            
            rewards_sum += reward[0]
            num_steps += 1
            
            if info[0]['timestep'] < 150:
                return_accuracies[i, info[0]['timestep']] = info[0]['val_acc']
                
        print("CUMULATIVE REWARD:", rewards_sum, "- NUM STEPS:", num_steps, "- VAL ACCURACY:", info[0]['val_acc'])
        rewards.append(rewards_sum)
        steps.append(num_steps)
        accuracies.append(info[0]['val_acc'])
        
    print("MEAN REWARD:", np.mean(rewards), "- MEAN STEPS:", np.mean(num_steps), "- MEAN ACCURACY:", np.mean(accuracies))
    
    return np.mean(return_accuracies, axis=0), np.min(num_steps)


def plot(arr, labels, y_lim=(0.0, 1.0)):
    for data, label in zip(arr, labels):
        plt.plot(np.arange(len(data)), data, label=label)

    plt.legend()
    plt.ylim(y_lim)
    plt.savefig('/opt/workspace/host_storage_hdd/results/basic_RL_results.png')
                

if __name__ == '__main__':
    min_steps_array = []
    
    env = make_vec_env(SelfTeachingEnv, n_envs=1, env_kwargs={'EPOCHS_PER_STEP': 2, 'ID': gpu_num, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0}) # , "HISTORY_LEN": 10, "HISTORY_MEAN": True})
    model = SAC(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [256, 512, 256, 128]})

    """
        # generate_expert_traj(expert_behaviour, 'expert_self_teaching', env, n_episodes=50)
        dataset = ExpertDataset(expert_path='/opt/workspace/host_storage_hdd/expert_self_teaching.npz', traj_limitation=-1, batch_size=64)

        model.pretrain(dataset, n_epochs=200)
        
        pretrain_accuracies, min_steps = test(model, env)
        min_steps_array.append(min_steps)
        
        model = SAC.load('/opt/workspace/host_storage_hdd/sac_self_teaching_decay_0.05.zip')    
        basic_trained_accuracies, min_steps = test(model, env)
        min_steps_array.append(min_steps)
        
        model = SAC.load('/opt/workspace/host_storage_hdd/best_sac_self_teaching_decay_0.05.zip')
        ep_limit_accuracies, min_steps = test(model, env)
        min_steps_array.append(min_steps)
        
        labeled_only_accuracies, min_steps = test(model, env, override_action=True)
        min_steps_array.append(min_steps)
        
        min_steps = np.min(min_steps_array)
        arr_to_plot = []
        arr_to_plot.append(pretrain_accuracies[:min_steps])
        arr_to_plot.append(basic_trained_accuracies[:min_steps])
        arr_to_plot.append(ep_limit_accuracies[:min_steps])
        arr_to_plot.append(labeled_only_accuracies[:min_steps])
        
        plot(arr_to_plot, labels=["pretrained", "RL trained - decay 0.05", "RL trained - best", "label only baseline"], y_lim=(0.8, 1.0))
        exit()
    """ 
    
    model.learn(total_timesteps=250000, log_interval=10, callback=learn_callback)
    model.save("/opt/workspace/host_storage_hdd/sac_self_teaching")

    test(model, env)
