import warnings
warnings.filterwarnings('ignore')

gpu_num = '2,3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

from datetime import datetime
import shutil
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from matplotlib import pyplot as plt

import torch
import torch.multiprocessing as mp

from multiprocessing import Process
from multiprocessing.managers import BaseManager

from env import SelfTeachingEnvV0 as SelfTeachingEnv
from sac_multi import SAC_Trainer, ReplayBuffer, share_parameters, worker

best_train_mean_episode_rewards = -np.inf
best_test_mean_episode_accuracy = -np.inf

def learn_callback(_locals, _globals):
    global best_train_mean_episode_rewards
    global best_test_mean_episode_accuracy
    
    if _locals['episode_reward'] == 0 and _locals['worker_id'] == 0:
        num_episodes = _locals['n_episodes'] + 1
        if _locals['step'] < _locals['learning_starts']:
            return True
 
        reward_lookback = 20
        # test_interval = int(5 + (1 - (['step'] - _locals['self'].learning_starts) / (_locals['total_timesteps'] - _locals['self'].learning_starts)) * 65)
        test_interval = 20
        test_episodes = 5
        
        mean_episode_rewards = np.mean(_locals['rewards'][-reward_lookback:])
            
        if mean_episode_rewards > best_train_mean_episode_rewards:
            best_train_mean_episode_rewards = mean_episode_rewards
            _locals['sac_trainer'].save_model(_locals['log_path'] + 'best_by_train_sac_self_teaching')
        
        if num_episodes % test_interval == 0:
            mean_accuracies, std_accuracies, mean_actions, std_actions = test(_locals['sac_trainer'], _locals['env'], n_episodes=test_episodes, scale=True)

            if mean_accuracies[-1] > best_test_mean_episode_accuracy:
                best_test_mean_episode_accuracy = mean_accuracies[-1]
                _locals['sac_trainer'].save_model(_locals['log_path'] + 'best_by_test_sac_self_teaching')

                plot_actions(mean_actions, std_actions, label=str(num_episodes) + "_test", color="C5", filepath=_locals['log_path'])
                plot([mean_accuracies], [std_accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.8, 1.0), filename=str(num_episodes) + "_test" + "_accuracy_%.4f" % (mean_accuracies[-1]), filepath=_locals['log_path'])
            
            if 'writer' in _locals:
                writer = _locals['writer']
                writer.add_scalar('Actions/meanTestActions', np.mean(mean_actions[0]), _locals['step'])
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


def test(model, env, n_episodes=10, override_action=False, scale=False, render=True):
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
            if render:
                env.render()

            if override_action:
                if isinstance(override_action, list):
                    action = torch.Tensor(override_action)
                else:
                    action = torch.ones(2).view((1, -1)) * -1   # np.ones(2).reshape((1, -1)) * -1
            else:
                action = model.policy_net.get_action(obs, deterministic=True)            
                    
            obs, reward, done, info = env.step(action)
            
            reward = reward.cpu().detach().numpy()
            info['val_acc'] = info['val_acc'].cpu().detach().numpy()
            action = action.view(-1)
            
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
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(100000)
    
    env = SelfTeachingEnv(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    
    mean_accs, std_accs = [], []
    
    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=rl_hidden_layer_sizes)
    sac_trainer.load_model(model_path + 'best_by_test_sac_self_teaching')
    sac_trainer.to_cuda()
    
    mean_acc, std_acc, mean_actions, std_actions = test(sac_trainer, env, override_action=[0.982, -7.0/9.0], n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    mean_acc, std_acc, mean_actions, std_actions = test(sac_trainer, env, override_action=True, n_episodes=10)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    mean_acc, std_acc, mean_actions, std_actions = test(sac_trainer, env, n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    """
        sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=[512, 1024, 512, 256])
        sac_trainer.load_model(model_path + 'best_by_train_sac_self_teaching')
        sac_trainer.to_cuda()
        mean_acc, std_acc, mean_actions, std_actions = test(sac_trainer, env, n_episodes=30)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        
        sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=[512, 1024, 512, 256])
        sac_trainer.load_model(model_path + 'final_sac_self_teaching')
        sac_trainer.to_cuda()
        mean_acc, std_acc, mean_actions, std_actions = test(sac_trainer, env, n_episodes=30)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
    """
    
    plot(mean_accs, std_accs, labels=["manually set thresholds", "label only baseline", "RL trained - test"], y_lim=(0.8, 1.0), filename=model_path.split('/')[-2], filepath='/opt/workspace/host_storage_hdd/results/')


def save_self(filepath):
    filepath = filepath + 'code/'
    os.makedirs(filepath, exist_ok=True)
    filenames = ['main.py', 'model.py', 'env.py', 'sac_multi.py', 'preprocess.py']
    
    for filename in filenames:
        shutil.copyfile(filename, filepath + filename)
    

if __name__ == '__main__':
    os.system('clear')

    env_kwargs = {'EPOCHS_PER_STEP': 2, 'N_TIMESTEPS': 150, "SIGNIFICANCE_DECAY": 0.0, 'N_CLUSTERS': 1}  # , 'REWARD_SCALE': 100}
    worker_offset = 0
    num_workers = 4
    initial_lr = 5e-5
    final_lr = 5e-5
    num_steps = 300000
    learning_starts = 10000
    batch_size = 16
    rl_hidden_layer_sizes = [64, 64]
    
    # test_pipeline('/opt/workspace/host_storage_hdd/results/2020-02-16_15-14-10.497811/')
    # exit()
    
    folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-")
    save_path = '/opt/workspace/host_storage_hdd/results/' + folder_name + '/'
    os.makedirs(save_path, exist_ok=True)
    save_self(save_path)
    
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(100000)

    env = SelfTeachingEnv(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    env.close()
    del env

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_layer_sizes=rl_hidden_layer_sizes, q_lr=initial_lr, pi_lr=initial_lr, alpha_lr=initial_lr, v_lr=initial_lr)

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
                                  'env_fn': SelfTeachingEnv,
                                  'env_kwargs': env_kwargs,
                                  'replay_buffer': replay_buffer,
                                  'num_steps': num_steps // num_workers,
                                  'learning_starts': learning_starts // num_workers,
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
        
    test_pipeline(save_path)

