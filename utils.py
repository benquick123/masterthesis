import os
import shutil
import collections
from multiprocessing.managers import BaseManager

import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import PIL

from sac_multi import ReplayBuffer

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
        test_interval = 5
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
                plot([mean_accuracies], [std_accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.5, 1.0), filename=str(num_episodes) + "_test" + "_accuracy_%.4f" % (mean_accuracies[-1]), filepath=_locals['log_path'])
            
            if 'writer' in _locals:
                writer = _locals['writer']
                writer.add_scalar('Actions/meanTestActions', np.mean(mean_actions[:, 0]), _locals['step'])
                writer.add_scalar('Accuracies/testAccuracies', mean_accuracies[-1], _locals['step'])
    
    return True


def test(model, env, n_episodes=10, override_action=False, scale=False, render=True, take_all_clusters=False):
    rewards = []
    steps = []
    actions = []
    
    return_accuracies = np.zeros((n_episodes, env.hyperparams['N_TIMESTEPS'] + 1))
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
                    action = torch.ones(env.action_space.shape) * -1
            else:
                action = model.policy_net.get_action(obs, deterministic=True)            
                    
            # env does not adhere do OpenAI spec anymore.
            # next_obs, obs, reward, done, info = env.step(action)
            obs, reward, done, info = env.step(action)
            
            reward = reward.cpu().detach().numpy()
            info['val_acc'] = info['val_acc'].cpu().detach().numpy()
            action = action.view(-1)
            
            rewards_sum += reward
            num_steps += 1
            
            if info['timestep'] < env.hyperparams['N_TIMESTEPS']:
                return_accuracies[i, info['timestep']] = info['val_acc']
            
            actions[-1].append(action.tolist())
                
        print(i, ": CUMULATIVE REWARD:", rewards_sum, "- NUM STEPS:", num_steps, "- VAL ACCURACY:", info['val_acc'])
        rewards.append(rewards_sum)
        steps.append(num_steps)
        return_accuracies[i, -1] = info['val_acc']
        
    print("MEAN REWARD:", np.mean(rewards), "- MEAN STEPS:", np.mean(num_steps), "- MEAN ACCURACY:", np.mean(return_accuracies[:, -1]))
    
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
                

def plot_actions(mean_actions, std_actions, label, color, filepath=None):
    plt.clf()
    for i in range(mean_actions.shape[1]):
        plt.plot(np.arange(len(mean_actions[:, i])), mean_actions[:, i], color=color)
        plt.fill_between(np.arange(len(mean_actions[:, i])), mean_actions[:, i] - std_actions[:, i], mean_actions[:, i] + std_actions[:, i], color=color, alpha=0.4)

    if filepath is not None:
        plt.savefig(filepath + label.replace(" ", "_") + "_actions" + ".svg")


def test_pipeline(env, trainer, model_path=None, save=True):
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    
    mean_accs, std_accs = [], []

    if model_path is not None:
        trainer.load_model(model_path + 'best_by_test_sac_self_teaching')
    trainer.to_cuda()
    
    mean_acc, std_acc, mean_actions, std_actions = test(trainer, env, override_action=[[0.982, -7.0/9.0]], n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    mean_acc, std_acc, mean_actions, std_actions = test(trainer, env, override_action=True, n_episodes=10)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    mean_acc, std_acc, mean_actions, std_actions = test(trainer, env, n_episodes=30)
    mean_accs.append(mean_acc)
    std_accs.append(std_acc)
    
    if save:
        filename = "test" if model_path is None else model_path.split("/")[-2]
        plot(mean_accs, std_accs, labels=["manually set thresholds", "label only baseline", "RL trained - test"], y_lim=(0.5, 1.0), filename=filename, filepath='/opt/workspace/host_storage_hdd/results/')


def save_self(filepath):
    filepath = filepath + 'code/'
    os.makedirs(filepath, exist_ok=True)
    filenames = ['main_basic.py', 'main_transfer.py', 'model.py', 'env.py', 'sac_multi.py', 'datasets.py']
    
    for filename in filenames:
        shutil.copyfile(filename, filepath + filename)


# code for elastic transforms from: https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Added correction: accepts only PIL.Image type.
    """
    # assert isinstance(image, PIL.Image)
    
    image = np.asarray(image)
    if len(image.shape) < 3:
        image = np.expand_dims(image, 2)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    
    if result.shape[2] == 1:
        result = result.reshape(result.shape[:2])
    return PIL.Image.fromarray(result.astype('uint8'))


class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma)