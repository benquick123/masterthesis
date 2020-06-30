from datetime import datetime
import errno
import os
import pickle
import re
import shutil
import collections
import importlib
from multiprocessing.managers import BaseManager

import seaborn
import torch
import torch.nn.functional as F
from torch import nn
import torchtext
import numpy as np
from matplotlib import pyplot as plt
import spacy
from sklearn.metrics import confusion_matrix

from datasets import TextDataset

best_train_mean_episode_rewards = -np.inf
best_test_mean_episode_accuracy = -np.inf


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    Code from: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class Logger:
    
    def __init__(self, save_path=None, backup_filenames=['main_basic.py', 'model.py', 'env.py', 'sac_multi.py', 'utils.py']):
        self.save_path = save_path
        self.file = None
        if self.save_path is not None:
            self.set_path(save_path=self.save_path)
            
        self.backup_filenames = backup_filenames
            
    def set_path(self, save_path=None, args=None):
        if save_path is not None:
            self.save_path = save_path
        elif args is not None:
            folder_name = str(datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
            self.save_path = os.path.join(args.results_folder, folder_name)
            self.save_path += "_" + args.from_dataset + "_to" if args.from_dataset != "" else ""
            self.save_path += "_" + args.dataset
            self.save_path += "_" + args.path_postfix if args.path_postfix != "" else ""
        else:
            raise AttributeError(errno.ENOENT, os.strerror(errno.ENOENT), "Either 'save_path' or 'args' must be set.")
        
        os.makedirs(self.save_path, exist_ok=True)
    
    def create_logdirs(self, args, save_self=True, save_config=True, save_args=True):
        # create folder name and save_path
        if self.save_path is None:
            self.set_path(args=args)
        
        # save code
        if save_self:
            self.save_code()
        
        # save experiment config
        if save_config:
            shutil.copyfile(os.path.join(args.config_path, args.dataset.lower() + ".json"), os.path.join(self.save_path, args.dataset.lower() + ".json"))
        
        # save arguments
        if save_args:
            f = open(os.path.join(self.save_path, "args"), "w")
            f.write(str(args).replace(", ", ",\n"))
            f.close()
            
        return self.save_path
    
    def save_code(self):
        filepath = os.path.join(self.save_path, 'code')
        os.makedirs(filepath, exist_ok=True)
        
        for filename in self.backup_filenames:
            shutil.copyfile(filename, os.path.join(filepath, filename))
    
    def print(self, *args, **kwargs):
        if self.save_path is not None:
            f = open(os.path.join(self.save_path, "log.log"), "a")
            print(str(datetime.now()) + ":", *args, **kwargs, file=f)
            f.close()
            
        print(str(datetime.now()) + ":", *args, **kwargs)


def learn_callback(_locals, _globals, reward_lookback=20, test_interval=100, test_episodes=5):
    global best_train_mean_episode_rewards
    global best_test_mean_episode_accuracy
    
    if _locals['n_episode_steps'] == 0 and _locals['worker_id'] == 0:
        num_episodes = _locals['n_episodes'] + 1
        if _locals['step'] < _locals['learning_starts']:
            return True
 
        mean_episode_rewards = np.mean(_locals['rewards'][-reward_lookback:])
            
        if mean_episode_rewards > best_train_mean_episode_rewards:
            best_train_mean_episode_rewards = mean_episode_rewards
            _locals['sac_trainer'].save_model(os.path.join(_locals['logger'].save_path, 'best_by_train_sac_self_teaching'))
        
        if num_episodes % test_interval == 0:
            mean_accuracies, std_accuracies, mean_actions, std_actions, _, _ = test(_locals['sac_trainer'], _locals['env'], _locals['logger'], n_episodes=test_episodes)
            
            if mean_accuracies[-1] > best_test_mean_episode_accuracy:
                best_test_mean_episode_accuracy = mean_accuracies[-1]
                _locals['sac_trainer'].save_model(os.path.join(_locals['logger'].save_path, 'best_by_test_sac_self_teaching'))

                plot_actions(mean_actions, std_actions, label=str(num_episodes) + "_test", color="C5", filepath=_locals['logger'].save_path)
                plot([mean_accuracies], [std_accuracies], labels=["n_episodes = " + str(num_episodes)], y_lim=(0.0, 1.0), filename=str(num_episodes) + "_test" + "_accuracy_%.4f" % (mean_accuracies[-1]), filepath=_locals['logger'].save_path)
            
            if 'writer' in _locals:
                writer = _locals['writer']
                writer.add_scalar('Actions/meanTestActions', np.mean(mean_actions[:, 0]), _locals['step'])
                writer.add_scalar('Accuracies/testAccuracies', mean_accuracies[-1], _locals['step'])
    
    return True


def test(model, env, logger=Logger(), n_episodes=10, override_action=False, render=True):
    rewards = []
    steps = []
    actions = []
    num_samples = []
    
    return_accuracies = np.zeros((n_episodes, env.hyperparams['max_timesteps'] + 1))
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        rewards_sum = 0
        num_steps = 0
        actions.append([])
        num_samples.append([])
        
        while not done:
            if render:
                env.render()

            if override_action:
                if isinstance(override_action, list):
                    action = torch.tensor(override_action)
                else:
                    action = torch.ones(env.action_space.shape) * -1
            else:
                action = model.policy_net.get_action(obs, deterministic=True)
                    
            obs, reward, done, info = env.step(action)
            
            reward = reward.cpu().detach().numpy()
            info['acc'] = info['acc'].cpu().detach().numpy()
            
            action = np.array([info['true_action'][i].cpu().detach().numpy() for i in range(len(info['true_action']))])
            
            rewards_sum += reward
            num_steps += 1
            
            if info['timestep'] < env.hyperparams['max_timesteps']:
                return_accuracies[i, info['timestep']] = info['acc']
            
            actions[-1].append(action.tolist())
            num_samples[-1].append(info['num_samples'].cpu().numpy() / len(env.X_unlabel))
                
        logger.print(i, ": CUMULATIVE REWARD:", rewards_sum, "- NUM STEPS:", num_steps, "- " + ("TEST" if env.is_testing else "VAL") + " ACCURACY:", info['acc'])
        rewards.append(rewards_sum)
        steps.append(num_steps)
        return_accuracies[i, -1] = info['acc']
        
    logger.print("MEAN REWARD:", np.mean(rewards), "- MEAN STEPS:", np.mean(num_steps), "- MEAN ACCURACY:", np.mean(return_accuracies[:, -1]))
    
    actions = np.array(actions)
    num_samples = np.array(num_samples)

    env.reset()
    return np.mean(return_accuracies, axis=0), np.std(return_accuracies, axis=0), np.mean(actions, axis=0), np.std(actions, axis=0), np.mean(num_samples, axis=0), np.std(num_samples, axis=0)


def plot(mean_arr, std_arr, labels, y_lim=(0.0, 1.0), filename='RL_results', filepath=None):
    plt.clf()
    for mean_data, std_data, label in zip(mean_arr, std_arr, labels):
        plt.plot(np.arange(len(mean_data)), mean_data, label=label)
        plt.fill_between(np.arange(len(std_data)), mean_data-std_data, mean_data+std_data, alpha=0.4)

    plt.legend()
    plt.ylim(y_lim)
    if filepath is not None:
        plt.savefig(os.path.join(filepath, filename + '.svg'))


def plot_actions(mean_actions, std_actions, label, color, filepath=None, y_lim=(0.0, 1.0)):
    plt.clf()
    for i in range(mean_actions.shape[1]):
        plt.plot(np.arange(len(mean_actions[:, i])), mean_actions[:, i], color=color)
        plt.fill_between(np.arange(len(mean_actions[:, i])), mean_actions[:, i] - std_actions[:, i], mean_actions[:, i] + std_actions[:, i], color=color, alpha=0.4)

    plt.ylim(y_lim)
    if filepath is not None:
        plt.savefig(os.path.join(filepath, label.replace(" ", "_") + "_actions" + ".svg"))


def plot_confusion_matrix(matrix, filepath=None):
    plt.figure(figsize=(6, 6))
    seaborn.heatmap(matrix)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    if filepath is not None:
        plt.savefig(os.path.join(filepath, "confusion_matrix.svg"))


def test_pipeline(env, trainer, logger=Logger(), model_path=None, all_samples_labeled=True, all_samples=True, manual_thresholds=True, labeled_samples=True, trained_model=True):
    env.is_testing = True
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    
    mean_accs, std_accs = [], []
    labels = []

    if model_path is not None:
        trainer.load_model(os.path.join(model_path, 'best_by_test_sac_self_teaching'))
    trainer.to_cuda()
    
    if all_samples_labeled:
        logger.print("all samples - labeled")
        env.known_labels = True
        tmp_alpha_lambda = env.hyperparams['unlabel_alpha']
        env.hyperparams['unlabel_alpha'] = lambda step : 1.0
        
        mean_acc, std_acc, _, _, _, _ = test(trainer, env, logger, override_action=[[0.0, 1.0]], n_episodes=10)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        labels.append("all samples - labeled")
        
        env.known_labels = False
        env.hyperparams['unlabel_alpha'] = tmp_alpha_lambda
        logger.print("Sanity check: unlabel_alpha =", env.hyperparams['unlabel_alpha'](0))

    if all_samples:
        logger.print("all samples - labeled & unlabeled")
        mean_acc, std_acc, _, _, _, _ = test(trainer, env, logger, override_action=[[0.0, 1.0]], n_episodes=10)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        labels.append("all samples - labeled & unlabeled")
        
    if manual_thresholds:
        logger.print("manually set thresholds")
        if isinstance(manual_thresholds, list):
            override_action = manual_thresholds
        else:
            override_action = [[0.982, -7.0/9.0]]
            
        mean_acc, std_acc, _, _, _, _ = test(trainer, env, logger, override_action=override_action, n_episodes=30)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        labels.append("manually set thresholds")

    if labeled_samples:
        logger.print("label only baseline")
        mean_acc, std_acc, _, _, _, _ = test(trainer, env, logger, override_action=True, n_episodes=10)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        labels.append("label only baseline")
    
    if trained_model:
        logger.print("RL trained - test")
        mean_acc, std_acc, mean_actions, std_actions, mean_samples, std_samples = test(trainer, env, logger, n_episodes=30)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        labels.append("RL trained - test")
    else:
        mean_actions, std_actions, mean_samples, std_samples = None, None, None, None
    
    if logger.save_path and any([all_samples, manual_thresholds, labeled_samples, trained_model]):
        if trained_model:
            plot_actions(mean_actions, std_actions, label="test", color="C5", filepath=logger.save_path)
            plot([mean_samples], [std_samples], labels=["num selected samples"], y_lim=(0, 1), filename='test_samples', filepath=logger.save_path)
        plot(mean_accs, std_accs, labels=labels, y_lim=(0.0, 1.0), filename="test_curves", filepath=logger.save_path)
        
    obs = env.reset()
    model = env.model
    for _ in range(env.hyperparams['max_timesteps']):
        a = trainer.policy_net.get_action(obs, deterministic=True)
        obs, _, done, _ = env.step(a)
        if done:
            break
    
    y_pred = model.predict(env.X_test, batch_size=env.hyperparams['pred_batch_size']).cpu().detach()
    cm = confusion_matrix(env.y_test.cpu().detach(), torch.argmax(y_pred, axis=1))
    
    if logger.save_path:
        filepath = "." if logger.save_path is None else logger.save_path
        plot_confusion_matrix(cm, filepath)
        
        # also save all the results to be pickable.
        pickle.dump({"mean_accs": mean_accs, "std_accs": std_accs, "labels": labels, "mean_actions": mean_actions, "std_actions": std_actions, "mean_samples": mean_samples, "std_samples": std_samples, "confusion_matrix": confusion_matrix}, 
                    open(os.path.join(logger.save_path, "test_results.pkl"), "wb"))
        
    env.is_testing = False


def imdb_preprocessing(path, loader_fn, text, target, train=True, emb_dim=100, max_sample_len=1000, load=True):
    try:
        if load:
            X, y = pickle.load(open(os.path.join(path, "tokenized_dataset_" + ("train" if train else "test") + "_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "rb"))
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
        train_dataset = dataset_loader_fn(os.path.join(path, "train"), text_field, label_field)
        test_dataset = None if train else dataset_loader_fn(os.path.join(path, "test"), text_field, label_field)

        vec = torchtext.vocab.GloVe(name='6B', dim=emb_dim, cache='/opt/workspace/host_storage_hdd/.vector_cache')
        text_field.build_vocab(train_dataset, vectors=vec, unk_init=torch.Tensor.normal_)
        label_field.build_vocab(train_dataset)
        
        dataset = train_dataset if train else test_dataset
        dataset = next(iter(torchtext.data.Iterator.splits((dataset, ), batch_sizes=(len(dataset), ), shuffle=False)[0]))
        
        X = getattr(dataset, text)
        y = getattr(dataset, target)

        pickle.dump((X, y), open(os.path.join(path, "tokenized_dataset_" + ("train" if train else "test") + "_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "wb"))
        if train:
            pickle.dump(text_field.vocab.vectors, open(os.path.join(path, "embed_vectors_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "wb"))

        return TextDataset(X, y)


def text_dataset_loader(path, dataset_key, train=True, emb_dim=100, max_sample_len=1000, load=True):
    # this function presumes a very specific loading structure; see below.
    try:
        if load:
            X, y = pickle.load(open(os.path.join(path, "tokenized_dataset_" + ("train" if train else "test") + "_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "rb"))
            return TextDataset(X, y)
        else:
            raise FileNotFoundError # raise exception to re-generate the data.
    except FileNotFoundError:
        from torchtext.utils import download_from_url, extract_archive
        from torchtext.datasets.text_classification import URLS, _csv_iterator, build_vocab_from_iterator
        dataset_tar = download_from_url(URLS[dataset_key], root="/".join(path.split("/")[:-1]))
        extract_archive(dataset_tar)
        
        train_path = os.path.join(path, "train.csv")
        train_dataset = np.array([(sample[0], list(sample[1])) for sample in _csv_iterator(train_path, ngrams=1, yield_cls=1)])
        
        test_path = None if train else os.path.join(path, "test.csv")
        test_dataset = None if train else np.array([(sample[0], list(sample[1])) for sample in _csv_iterator(train_path, ngrams=1, yield_cls=1)])

        vec = torchtext.vocab.GloVe(name='6B', dim=emb_dim, cache='/opt/workspace/host_storage_hdd/.vector_cache')
        vocab = build_vocab_from_iterator(train_dataset[:, 1])
        vocab.set_vectors(vec.stoi, vec.vectors, emb_dim, unk_init=torch.Tensor.normal_)
        
        X = []
        y = []
        dataset = train_dataset if train else test_dataset
        for target, data in dataset:
            sample_x = []
            for token in data:
                sample_x.append(vocab.stoi[token])
                
            if len(sample_x) < max_sample_len:
                sample_x += [vocab.stoi["<pad>"]] * (max_sample_len - len(sample_x))
            else:
                sample_x = sample_x[:max_sample_len]
                
            X.append(sample_x)
            y.append(target)

        X = torch.tensor(X)
        y = torch.tensor(y)
        
        pickle.dump((X, y), open(os.path.join(path, "tokenized_dataset_" + ("train" if train else "test") + "_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "wb"))
        if train:
            pickle.dump(vocab.vectors, open(os.path.join(path, "embed_vectors_" + str(max_sample_len) + "_" + str(emb_dim) + ".pkl"), "wb"))
        
        return TextDataset(X, y)
