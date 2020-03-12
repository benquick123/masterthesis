import os
import pickle
import re
import shutil
import collections
import importlib
from multiprocessing.managers import BaseManager

import seaborn
import torch
import torchtext
import numpy as np
from matplotlib import pyplot as plt
import spacy
from sklearn.metrics import confusion_matrix

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
    
    return_accuracies = np.zeros((n_episodes, env.hyperparams['max_timesteps'] + 1))
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
            
            if info['timestep'] < env.hyperparams['max_timesteps']:
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


def plot_confusion_matrix(matrix, filepath=None):
    plt.figure(figsize=(6, 6))
    seaborn.heatmap(matrix)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    if filepath is not None:
        plt.save(os.path.join(filepath, "confusion_matrix.svg"))


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
        plot(mean_accs, std_accs, labels=["manually set thresholds", "label only baseline", "RL trained - test"], y_lim=(0.5, 1.0), filename="test_curves", filepath=model_path)
        
    model = env.model
    y_pred = model.predict(env.X_test, batch_size=env.hyperparams['pred_batch_size'])
    cm = confusion_matrix(env.y_test, y_pred)   
    
    if save:
        filepath = "." if model_path is None else model_path
        plot_confusion_matrix(cm, filepath)


def save_self(filepath):
    filepath = filepath + 'code/'
    os.makedirs(filepath, exist_ok=True)
    filenames = ['main_basic.py', 'main_transfer.py', 'model.py', 'env.py', 'sac_multi.py', 'datasets.py']
    
    for filename in filenames:
        shutil.copyfile(filename, filepath + filename)


def text_preprocessing(path, loader_fn, text, target, train=True, emb_dim=100, max_sample_len=1000, load=True):
    try:
        if load:
            X, y = pickle.load(open(os.path.join(path, "tokenized_dataset_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "rb"))
            return TextDataset(X, y)
        else:
            raise FileNotFoundError # raise exception to re-generate the data.
    except FileNotFoundError:
        nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        
        def clean(text):
            text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove all non-alphanumeric characters.
            text = re.sub(r'https?:/\/\S+', ' ', text) # remove links.
            return text.strip()
        
        def tokenizer(s):
            return [w.text.lower() for w in nlp(clean(s))]
        
        module_path = loader_fn.split(".")
        dataset_loader_fn = getattr(importlib.import_module(".".join(module_path[:-1])), module_path[-1])
        
        text_field = torchtext.data.Field(tokenize=tokenizer, fix_length=max_sample_len, batch_first=True)
        label_field = torchtext.data.LabelField(dtype=torch.float)
        train_vs_test = "train" if train else "test"
        dataset = dataset_loader_fn(os.path.join(path, train_vs_test), text_field, label_field)
        
        vec = torchtext.vocab.GloVe(name='6B', dim=emb_dim, cache='/opt/workspace/host_storage_hdd/.vector_cache')
        # text_field.build_vocab(dataset)
        text_field.build_vocab(dataset, vectors=vec)
        label_field.build_vocab(dataset)
        
        dataset = next(iter(torchtext.data.Iterator.splits((dataset, ), batch_sizes=(len(dataset), ))[0]))
        
        X = getattr(dataset, text)
        y = getattr(dataset, target)
        
        pickle.dump((X, y), open(os.path.join(path, "tokenized_dataset_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "wb"))

        return TextDataset(X, y)
        
        
class TextDataset(torch.utils.data.TensorDataset):
    
    def __init__(self, X, y, **kwargs):
        super(TextDataset, self).__init__(X, y, **kwargs)
        self.data = X
        self.targets = y