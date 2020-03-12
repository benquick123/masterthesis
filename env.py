import gym
import pickle
import json
import os
import importlib
# from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from datetime import datetime

import numpy as np # only for dataset construction

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss
from torch import Tensor
from torchvision import transforms

from hidden_features import train_autoencoder, cluster_images
from model import DenseModel

gym.register("SelfTeaching-v0")
gym.register("SelfTeaching-v1")
gym.register("SelfTeachingBase-v0")

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum().double() / y_true.shape[0]


class SelfTeachingEnvV0(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v0")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, significance_threshold=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, REWARD_SCALE=10, BASE_HIDDEN_LAYER_SIZES=[256, 128], IMAGE_TRANSFORMS=None):
        super(SelfTeachingEnvV0, self).__init__()
        
        print("Initializing environment.")
        self.hyperparams = dict(locals())
        del self.hyperparams['self']
        del self.hyperparams['__class__']
                
        assert 'N_CLUSTERS' in self.hyperparams and self.hyperparams['N_CLUSTERS'] > 0
        assert 'Y_ESTIMATED_LR' in self.hyperparams and 0.0 < self.hyperparams['Y_ESTIMATED_LR'] < 1.0
        assert 'N_TIMESTEPS' in self.hyperparams and self.hyperparams['N_TIMESTEPS'] > 0
        assert 'BATCH_SIZE' in self.hyperparams
        assert 'PRED_BATCH_SIZE' in self.hyperparams
        assert 'significance_threshold' in self.hyperparams
        assert 'SIGNIFICANCE_DECAY' in self.hyperparams and 0.0 <= self.hyperparams['SIGNIFICANCE_DECAY'] <= 1.0
        assert 'MIN_TIMESTEPS' in self.hyperparams
        assert 'EPOCHS_PER_STEP' in self.hyperparams
        assert 'ID' in self.hyperparams
        assert 'REWARD_SCALE' in self.hyperparams and self.hyperparams['REWARD_SCALE'] > 0.0
        assert 'BASE_HIDDEN_LAYER_SIZES' in self.hyperparams and isinstance(self.hyperparams['BASE_HIDDEN_LAYER_SIZES'], list)
        assert 'IMAGE_TRANSFORMS' in self.hyperparams and (self.hyperparams['IMAGE_TRANSFORMS'] is None or isinstance(self.hyperparams['IMAGE_TRANSFORMS'], transforms.Compose))
        
        self.hyperparams['GPU_ID'] = self.hyperparams['ID'] % torch.cuda.device_count()
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.y_train.shape[1] ** 2 + 5, )))
    
    def _generate_data(self, save=True):
        print("No data exist. Generating.")
        
        mnist_train = MNIST('/opt/workspace/host_storage_hdd', download=True, train=True)
        mnist_test = MNIST('/opt/workspace/host_storage_hdd', download=True, train=False)
        X_train = mnist_train.data.numpy()
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy()
        y_test = mnist_test.targets.numpy()
        
        # reshape, cast and scale in one step.
        X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
        X_train, self.X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
        self.X_train, self.X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
        
        self.X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
        
        num_classes = len(np.unique(y_train))
        to_categorical = lambda y, n : np.eye(n, dtype=y.dtype)[y]

        self.y_train = to_categorical(y_train, num_classes)
        self.y_test = to_categorical(y_test, num_classes)
        self.y_val = to_categorical(y_val, num_classes)
        
        _, encoder = train_autoencoder(self.X_train)
        groups, hidden_representations, self.group_centers = cluster_images(np.concatenate([self.X_unlabel, self.X_val], axis=0), encoder, n_clusters=self.hyperparams['N_CLUSTERS'], plot=False)
        self.X_unlabel_groups, self.X_unlabel_hidden = groups[:len(self.X_unlabel)], hidden_representations[:len(self.X_unlabel)]
        self.X_val_groups, self.X_val_hidden = groups[len(self.X_unlabel):], hidden_representations[len(self.X_unlabel):]
        
        if save:
            f = open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(self.hyperparams['N_CLUSTERS']) + '.pickle', 'wb')
            pickle.dump({
                'X_train': self.X_train,
                'y_train': self.y_train,
                'X_unlabel': self.X_unlabel,
                'X_unlabel_groups': self.X_unlabel_groups,
                'X_unlabel_hidden': self.X_unlabel_hidden,
                'X_val': self.X_val,
                'y_val': self.y_val.cuda(self.hyperparams['GPU_ID']),
                'X_val_groups': self.X_val_groups,
                'X_val_hidden': self.X_val_hidden,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'group_centers': self.group_centers,
            }, f)
            f.close()
    
    def _load_data(self):
        print("Loading data.")
        try:
            data = pickle.load(open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(self.hyperparams['N_CLUSTERS']) + '.pickle', 'rb'))
        except FileNotFoundError:
            self._generate_data(save=True)
            
        data = pickle.load(open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(self.hyperparams['N_CLUSTERS']) + '.pickle', 'rb'))
        self.X_train = torch.Tensor(data['X_train'])
        self.y_train = torch.Tensor(data['y_train'])
        self.X_unlabel = torch.Tensor(data['X_unlabel'])
        # self.X_unlabel_groups = torch.Tensor(data['X_unlabel_groups']).cuda(self.hyperparams['GPU_ID'])
        self.X_val = torch.Tensor(data['X_val'])
        self.y_val = torch.Tensor(data['y_val'])
        # self.X_val_groups = torch.Tensor(data['X_val_groups']).cuda(self.hyperparams['GPU_ID'])
        self.X_test = torch.Tensor(data['X_test'])
        self.y_test = torch.Tensor(data['y_test'])
        # self.group_centers = torch.Tensor(data['group_centers']).cuda(self.hyperparams['GPU_ID'])
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            self.model = DenseModel(self.X_train.shape[1], self.y_train.shape[1], layer_sizes=self.hyperparams['BASE_HIDDEN_LAYER_SIZES']).cuda(self.hyperparams['GPU_ID'])
            self.model.reset()
            self.model_optimizer = Adam(self.model.parameters(), lr=0.001)
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
            
    def _get_state(self):
        state = torch.zeros(self.observation_space.shape)
        y_val_argmax = torch.argmax(self.y_val, axis=1)
        last_y_val_pred_exp = self.last_y_val_pred.exp()
        
        for i in range(self.y_val.shape[1]):
            mask = i == y_val_argmax
            state[i * self.y_val.shape[1]:(i+1) * self.y_val.shape[1]] = last_y_val_pred_exp[mask].mean(axis=0)
            
        state = state.view(-1)
        state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
        return state
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['SIGNIFICANCE_DECAY']) * self.reward_moving_average + self.hyperparams['SIGNIFICANCE_DECAY'] * reward
        if self.timestep < self.hyperparams['MIN_TIMESTEPS']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['significance_threshold']
            
    def step(self, action):
        # rescale action to [0, 1] range.
        action = ((action + 1) / 2).view(-1).cpu()
        range_scale = 0.5 - torch.abs(0.5 - action[0])
        action[1] = action[1] * range_scale
        self.last_action = [action[0] - action[1], action[0] + action[1]]
        
        tau1, tau2 = self.last_action[0], self.last_action[1]

        assert tau1 <= tau2
        
        y_unlabel_estimates_max, y_unlabel_estimates_argmax = torch.max(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max > tau1) & (y_unlabel_estimates_max < tau2)
        self.len_selected_samples = y_unlabel_estimates_indices.sum().double()
    
        y_pred_binary = torch.zeros(self.last_y_unlabel_pred.shape).scatter_(1, y_unlabel_estimates_argmax.view((-1, 1)), 1.0)
        
        self.model.fit(torch.cat((self.X_train, self.X_unlabel[y_unlabel_estimates_indices]), axis=0),
                        torch.cat((self.y_train, y_pred_binary[y_unlabel_estimates_indices]), axis=0),
                        optimizer=self.model_optimizer,
                        loss_fn=self.model_loss,
                        epochs=self.hyperparams['EPOCHS_PER_STEP'], 
                        batch_size=self.hyperparams['BATCH_SIZE'], 
                        verbose=0,
                        gpu_id=self.hyperparams['GPU_ID'],
                        transforms=self.hyperparams['IMAGE_TRANSFORMS'])
        
        self.last_y_val_pred = self.model.predict(self.X_val).detach().cpu()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel).detach().cpu()
        self.last_y_label_pred = self.model.predict(self.X_train).detach().cpu()
        
        self.y_unlabel_estimates = (1 - self.hyperparams['Y_ESTIMATED_LR']) * self.y_unlabel_estimates + self.hyperparams['Y_ESTIMATED_LR'] * self.last_y_unlabel_pred.exp()
        self.y_unlabel_estimates /= self.y_unlabel_estimates.sum(axis=1).view((-1, 1))

        new_accuracy = accuracy_score(torch.argmax(self.y_val, axis=1), torch.argmax(self.last_y_val_pred, axis=1))
        
        self.last_reward = new_accuracy - self.last_val_accuracy    # -(new_mse_loss - self.last_mse_loss)    # new_accuracy - self.last_val_accuracy
        reward_scale_factor = self.hyperparams['REWARD_SCALE']
        self.last_reward *= reward_scale_factor
        
        self.last_val_accuracy = new_accuracy
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val, axis=1))
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1))
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1))
        
        self.last_state = self._get_state()
   
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['N_TIMESTEPS'] or not self._significant(self.last_reward) else False
            
        return self.last_state, self.last_reward, terminal, { 'val_acc': self.last_val_accuracy, 'timestep': self.timestep }
        
    def reset(self):
        self._initialize_model()
        
        self.last_action = torch.zeros(self.action_space.shape[0])
        
        self.last_y_val_pred = self.model.predict(self.X_val).detach().cpu()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel).detach().cpu()
        self.last_y_label_pred = self.model.predict(self.X_train).detach().cpu()
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.y_train.shape[1])) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        # self.last_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['significance_threshold']
        
        self.last_val_accuracy = accuracy_score(torch.argmax(self.y_val, axis=1), torch.argmax(self.last_y_val_pred, axis=1))
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1))
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val, axis=1))
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1))
        
        self.last_state = self._get_state()

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nThresholds:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        render_string += "\nState:\n" + str(self.last_state[:-5].view((self.y_val.shape[1], self.y_val.shape[1])).detach().numpy())[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['ID']) + ".log", "w"))
        
        return render_string


class SelfTeachingEnvV1(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v1")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, significance_threshold=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, REWARD_SCALE=10):
        super(SelfTeachingEnvV1, self).__init__()
        
        print("Initializing environment.")
        self.hyperparams = dict(locals())
        del self.hyperparams['self']
        del self.hyperparams['__class__']
                
        assert 'N_CLUSTERS' in self.hyperparams and self.hyperparams['N_CLUSTERS'] > 0
        assert 'Y_ESTIMATED_LR' in self.hyperparams and 0.0 < self.hyperparams['Y_ESTIMATED_LR'] <= 1.0
        assert 'N_TIMESTEPS' in self.hyperparams and self.hyperparams['N_TIMESTEPS'] > 0
        assert 'BATCH_SIZE' in self.hyperparams
        assert 'PRED_BATCH_SIZE' in self.hyperparams
        assert 'significance_threshold' in self.hyperparams
        assert 'SIGNIFICANCE_DECAY' in self.hyperparams and 0.0 <= self.hyperparams['SIGNIFICANCE_DECAY'] <= 1.0
        assert 'MIN_TIMESTEPS' in self.hyperparams
        assert 'EPOCHS_PER_STEP' in self.hyperparams
        assert 'ID' in self.hyperparams
        assert 'REWARD_SCALE' in self.hyperparams and self.hyperparams['REWARD_SCALE'] > 0.0
        
        self.hyperparams['GPU_ID'] = self.hyperparams['ID'] % torch.cuda.device_count()
        
        self.model = None
        
        self._load_data()
        
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 * self.hyperparams['N_CLUSTERS'], ))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ))
        # self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.y_train.shape[1] * self.hyperparams['N_CLUSTERS'] + 5, )))
        # self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.y_train.shape[1] ** 2 + 5, )))
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((4 * self.y_train.shape[1] + 7, )))
    
    def _generate_data(self, save=True):
        print("No data exist. Generating.")
        
        mnist_train = MNIST('/opt/workspace/host_storage_hdd', download=True, train=True)
        mnist_test = MNIST('/opt/workspace/host_storage_hdd', download=True, train=False)
        X_train = mnist_train.data.numpy()
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy()
        y_test = mnist_test.targets.numpy()
        
        # reshape, cast and scale in one step.
        X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
        X_train, self.X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025, random_state=0)
        self.X_train, self.X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99, random_state=0)
        
        self.X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
        
        num_classes = len(np.unique(y_train))
        to_categorical = lambda y, n : np.eye(n, dtype=y.dtype)[y]

        self.y_train = to_categorical(y_train, num_classes)
        self.y_test = to_categorical(y_test, num_classes)
        self.y_val = to_categorical(y_val, num_classes)
        
        _, encoder = train_autoencoder(self.X_train)
        groups, hidden_representations, self.group_centers = cluster_images(np.concatenate([self.X_unlabel, self.X_val], axis=0), encoder, n_clusters=self.hyperparams['N_CLUSTERS'], plot=False)
        self.X_unlabel_groups, self.X_unlabel_hidden = groups[:len(self.X_unlabel)], hidden_representations[:len(self.X_unlabel)]
        self.X_val_groups, self.X_val_hidden = groups[len(self.X_unlabel):], hidden_representations[len(self.X_unlabel):]
        
        if save:
            f = open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(self.hyperparams['N_CLUSTERS']) + '.pickle', 'wb')
            pickle.dump({
                'X_train': self.X_train,
                'y_train': self.y_train,
                'X_unlabel': self.X_unlabel,
                'X_unlabel_groups': self.X_unlabel_groups,
                'X_unlabel_hidden': self.X_unlabel_hidden,
                'X_val': self.X_val,
                'y_val': self.y_val.cuda(self.hyperparams['GPU_ID']),
                'X_val_groups': self.X_val_groups,
                'X_val_hidden': self.X_val_hidden,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'group_centers': self.group_centers,
            }, f)
            f.close()
    
    def _load_data(self):
        print("Loading data.")
        try:
            data = pickle.load(open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(self.hyperparams['N_CLUSTERS']) + '.pickle', 'rb'))
            self.X_train = torch.Tensor(data['X_train']).cuda(self.hyperparams['GPU_ID'])
            self.y_train = torch.Tensor(data['y_train']).cuda(self.hyperparams['GPU_ID'])
            self.X_unlabel = torch.Tensor(data['X_unlabel']).cuda(self.hyperparams['GPU_ID'])
            self.X_unlabel_groups = torch.Tensor(data['X_unlabel_groups']).cuda(self.hyperparams['GPU_ID']).long()
            self.X_val = torch.Tensor(data['X_val']).cuda(self.hyperparams['GPU_ID'])
            self.y_val = torch.Tensor(data['y_val']).cuda(self.hyperparams['GPU_ID'])
            self.X_val_groups = torch.Tensor(data['X_val_groups']).cuda(self.hyperparams['GPU_ID']).long()
            self.X_test = torch.Tensor(data['X_test']).cuda(self.hyperparams['GPU_ID'])
            self.y_test = torch.Tensor(data['y_test']).cuda(self.hyperparams['GPU_ID'])
            self.group_centers = torch.Tensor(data['group_centers']).cuda(self.hyperparams['GPU_ID'])
        except FileNotFoundError:
            self._generate_data(save=True)
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            self.model = DenseModel(self.X_train.shape[1], self.y_train.shape[1]).cuda(self.hyperparams['GPU_ID'])
            self.model.reset()
            self.model_optimizer = Adam(self.model.parameters(), lr=0.001)
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
            
    def _get_state(self, group=0):
        state = torch.zeros(self.observation_space.shape).cuda(self.hyperparams['GPU_ID'])
        mask = group == self.X_val_groups
        last_y_val_pred_exp_masked = self.last_y_val_pred.exp()[mask]
        last_y_val_argmax_masked = torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1)[mask]
        
        state[:self.y_train.shape[1]] = self.y_val[mask].mean(axis=0)
        
        state[self.y_train.shape[1]:2*self.y_train.shape[1]] = last_y_val_pred_exp_masked.mean(axis=0)
        
        for i in range(self.y_train.shape[1]):
            try:
                class_mask = i == last_y_val_argmax_masked
                state[2*self.y_train.shape[1]+i] = accuracy_score(last_y_val_argmax_masked[class_mask], torch.argmax(last_y_val_pred_exp_masked[class_mask], axis=1))
            except RuntimeError:
                state[2*self.y_train.shape[1]+i] = 0.0
        
        state[3*self.y_train.shape[1]:4*self.y_train.shape[1]] = last_y_val_pred_exp_masked.std(axis=0)
        
        state[4*self.y_train.shape[1]:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], 
                                                        self.last_train_accuracy, self.last_train_loss, 
                                                        self.last_val_accuracy, self.last_val_loss,
                                                        self.last_group_accuracy, self.last_group_loss])
        
        assert state.shape[0] == self.observation_space.shape[0]
        return state
        
        """
            state = torch.zeros(self.observation_space.shape).cuda(self.hyperparams['GPU_ID'])
            y_val_argmax = torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1)
            last_y_val_pred_exp = self.last_y_val_pred.exp()
            
            for i in range(self.y_val.shape[1]):
                mask = i == y_val_argmax
                state[i * self.y_val.shape[1]:(i+1) * self.y_val.shape[1]] = last_y_val_pred_exp[mask].mean(axis=0)
                
            state = state.view(-1)
            state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
            return state
        """
        
        """
            state = torch.zeros(self.observation_space.shape).cuda(self.hyperparams['GPU_ID'])
            last_y_val_pred_exp = self.last_y_val_pred.exp()
            
            for i in range(self.hyperparams['N_CLUSTERS']):
                mask = i == self.X_val_groups
                state[i * self.y_val.shape[1]:(i+1) * self.y_val.shape[1]] = ((self.y_val[mask] - last_y_val_pred_exp[mask]) ** 2).mean(axis=0)
            
            state = state.view(-1)
            state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
            return state
        """
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['SIGNIFICANCE_DECAY']) * self.reward_moving_average + self.hyperparams['SIGNIFICANCE_DECAY'] * reward
        if self.timestep < self.hyperparams['MIN_TIMESTEPS']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['significance_threshold']
            
    def step(self, action):
        # rescale action from [-1, 1] to [0, 1] range.
        """
            action = ((action + 1) / 2).view((-1, 2))
            range_scale = 0.5 - torch.abs(0.5 - action[:, 0])
            action[:, 1] = action[:, 1] * range_scale
            self.last_action = torch.zeros(action.shape).cuda(self.hyperparams['GPU_ID'])
            self.last_action[:, 0] = action[:, 0] - action[:, 1]
            self.last_action[:, 1] = action[:, 0] + action[:, 1]

            assert torch.all(self.last_action[:, 0] <= self.last_action[:, 1])

            tau = torch.index_select(self.last_action, 0, self.X_unlabel_groups)
        """
        action = ((action + 1) / 2).view((-1, ))
        range_scale = 0.5 - torch.abs(0.5 - action[0])
        action[1] = action[1] * range_scale
        
        self.last_action = torch.zeros(action.shape).cuda(self.hyperparams['GPU_ID'])
        self.last_action[0] = action[0] - action[1]
        self.last_action[1] = action[0] + action[1]
        
        mask = self.last_group == self.X_unlabel_groups
        y_unlabel_estimates_masked_max, y_unlabel_estimates_masked_argmax = torch.max(self.y_unlabel_estimates[mask], axis=1)
        y_unlabel_estimates_masked_indices = (y_unlabel_estimates_masked_max > self.last_action[0]) & (y_unlabel_estimates_masked_max < self.last_action[1])
        self.len_selected_samples = y_unlabel_estimates_masked_indices.sum().double()
        
        y_pred_binary = torch.zeros(self.last_y_unlabel_pred[mask].shape).cuda(self.hyperparams['GPU_ID']).scatter_(1, y_unlabel_estimates_masked_argmax.view((-1, 1)), 1.0)
        
        self.model.fit(torch.cat((self.X_train, self.X_unlabel[mask][y_unlabel_estimates_masked_indices]), axis=0),
                        torch.cat((self.y_train, y_pred_binary[y_unlabel_estimates_masked_indices]), axis=0),
                        optimizer=self.model_optimizer,
                        loss_fn=self.model_loss,
                        epochs=self.hyperparams['EPOCHS_PER_STEP'], 
                        batch_size=self.hyperparams['BATCH_SIZE'], 
                        verbose=0,
                        gpu_id=self.hyperparams['GPU_ID'])
        
        self.last_y_val_pred = self.model.predict(self.X_val)
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel)
        self.last_y_label_pred = self.model.predict(self.X_train)
        
        self.y_unlabel_estimates = (1 - self.hyperparams['Y_ESTIMATED_LR']) * self.y_unlabel_estimates + self.hyperparams['Y_ESTIMATED_LR'] * self.last_y_unlabel_pred.exp()
        self.y_unlabel_estimates /= self.y_unlabel_estimates.sum(axis=1).view((-1, 1))

        new_accuracy = accuracy_score(torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1), torch.argmax(self.last_y_val_pred, axis=1))
        # new_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        
        self.last_reward = new_accuracy - self.last_val_accuracy    # -(new_mse_loss - self.last_mse_loss)
        reward_scale_factor = self.hyperparams['REWARD_SCALE']
        self.last_reward *= reward_scale_factor
        # self.last_mse_loss = new_mse_loss
        
        self.last_val_accuracy = new_accuracy
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1))
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1))
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1))
        self.last_group_accuracy = accuracy_score(torch.argmax(self.y_val[self.last_group == self.X_val_groups], axis=1), torch.argmax(self.last_y_val_pred[self.last_group == self.X_val_groups], axis=1))
        self.last_group_loss = self.nll_loss(self.last_y_val_pred[self.last_group == self.X_val_groups], torch.argmax(self.y_val[self.last_group == self.X_val_groups], axis=1))
        
        self.last_next_state = self._get_state(group=self.last_group)
        self.last_group = torch.randint(0, self.hyperparams['N_CLUSTERS'], (1, ))[0].cuda(self.hyperparams['GPU_ID'])
        self.last_new_state = self._get_state(group=self.last_group)
   
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['N_TIMESTEPS'] or not self._significant(self.last_reward) else False
            
        return self.last_next_state, self.last_new_state, self.last_reward, terminal, { 'val_acc': self.last_val_accuracy, 'timestep': self.timestep }
        
    def reset(self):
        self._initialize_model()
        
        self.last_action = torch.zeros(self.action_space.shape[0]).cuda(self.hyperparams['GPU_ID']) # .view((-1, 2))
        
        self.last_y_val_pred = self.model.predict(self.X_val).detach()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel).detach()
        self.last_y_label_pred = self.model.predict(self.X_train).detach()
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.y_train.shape[1])).cuda(self.hyperparams['GPU_ID']) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        # self.last_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['significance_threshold']
        
        self.last_group = torch.randint(0, self.hyperparams['N_CLUSTERS'], (1, ))[0].cuda(self.hyperparams['GPU_ID'])
        
        self.last_val_accuracy = accuracy_score(torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1), torch.argmax(self.last_y_val_pred, axis=1)).detach()
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val.cuda(self.hyperparams['GPU_ID']), axis=1)).detach()
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1)).detach()
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1)).detach()
        self.last_group_accuracy = accuracy_score(torch.argmax(self.y_val[self.last_group == self.X_val_groups], axis=1), torch.argmax(self.last_y_val_pred[self.last_group == self.X_val_groups], axis=1))
        self.last_group_loss = self.nll_loss(self.last_y_val_pred[self.last_group == self.X_val_groups], torch.argmax(self.y_val[self.last_group == self.X_val_groups], axis=1))
        
        self.last_new_state = self._get_state()
        self.last_next_state = torch.zeros(self.observation_space.shape)
        
        return self.last_new_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        # render_string += "\n\nState & action:\n"
        # for state_part, action_part in zip(self.last_state[:-5].view((self.hyperparams['N_CLUSTERS'], self.y_val.shape[1])).cpu().detach().numpy(), self.last_action.cpu().detach().numpy()):
        """tmp_state = self.last_state[:-7].view((self.y_val.shape[1], self.y_val.shape[1]))
        for i in range(max(self.y_val.shape[1] + 1, self.hyperparams['N_CLUSTERS'])):       # zip(self.last_state[:-5].view((self.hyperparams['N_CLUSTERS'], self.y_val.shape[1])).cpu().detach().numpy(), self.last_action.cpu().detach().numpy()):
            # render_string += str(state_part) + "    \t" + str(action_part) + "\n"
            if i < tmp_state.shape[0]:
                render_string += str(tmp_state[i].cpu().detach().numpy()) + " "
            elif i == tmp_state.shape[0]:
                render_string += str(['%.3f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "") + "\t\t\t\t      "
            else:
                render_string += str(" " * len(str(tmp_state[-1].cpu().detach().numpy()))) + " "
            
            if i < self.last_action.shape[0]:
                render_string += str(self.last_action[i].cpu().detach().numpy()) + "\n"
            else:
                render_string += "\n"
        """
        render_string += "\nLast group: %d" % (self.last_group)
        render_string += "\n\nThresholds:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        render_string += "\nState:\n" + str(self.last_new_state[:-7].view((-1, self.y_val.shape[1])).cpu().detach().numpy())[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_new_state[-7:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['ID']) + ".log", "w"))
        
        return render_string


class SelfTeachingBaseEnv(gym.Env):
    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeachingBase-v0")
    
    def __init__(self, 
                 dataset, 
                 config_path="./config", 
                 default_hyperparams={"y_estimated_lr": 0.3,
                                      "max_timesteps": 500,
                                      "min_timesteps": 0,
                                      "batch_size": 64,
                                      "pred_batch_size": -1,
                                      "significance_threshold": 0.001,
                                      "significance_decay": 0.0,
                                      "epochs_per_step": 1,
                                      "worker_id": 0,
                                      "reward_scale": 1.0,
                                      "model_lr": 0.001,
                                      "train_size": 500,
                                      "val_size": 1500,
                                      "loader_kwargs": {
                                          "root": "/opt/workspace/host_storage_hdd",
                                          "download": True
                                          }
                                      }, 
                 override_hyperparams={}):
        
        super(SelfTeachingBaseEnv, self).__init__()
        
        json_config = json.load(open(os.path.join(config_path, dataset.lower() + ".json"), "r"))
        # take care for correct hyperparameters initialization
        self.hyperparams = dict(default_hyperparams)
        self.hyperparams.update(json_config)
        self.hyperparams.update(override_hyperparams)
        
        self.hyperparams['gpu_id'] = self.hyperparams['worker_id'] % torch.cuda.device_count()
        print("Initializing environment:", self.hyperparams['worker_id'])
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.hyperparams['n_classes'] ** 2 + 5, )))
    
    def _load_data(self):
        print("Loading data:", self.hyperparams['name'])
        module_path = self.hyperparams['loader_fn'].split(".")
        dataset_loader_fn = getattr(importlib.import_module(".".join(module_path[:-1])), module_path[-1])
        
        assert "train" not in self.hyperparams['loader_kwargs']
        # implementation assumes a interface otherwise found in torchvision.datasets.
        train = dataset_loader_fn(train=True, **self.hyperparams['loader_kwargs'])
        test = dataset_loader_fn(train=False, **self.hyperparams['loader_kwargs'])
        
        # ensure train and test are numpy arrays by first converting them to Tensor and then back to np.array
        self.X_train = torch.tensor(train.data).numpy()
        self.y_train = torch.tensor(train.targets).numpy()
        self.X_test = torch.tensor(test.data).numpy()
        self.y_test = torch.tensor(test.targets).numpy()
        
        if isinstance(self.hyperparams['train_size'], float):
            self.hyperparams['train_size'] = int(len(X_train) * self.hyperparams['train_size'])
        if isinstance(self.hyperparams['val_size'], float):
            self.hyperparams['val_size'] = int(len(X_train) * self.hyperparams['val_size'])
        
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        if np.max(self.X_train) > 1.0:
            self.X_train = self.X_train / 255
            self.X_test = self.X_test / 255
            
        if self.hyperparams['model_init_fn'] == 'model.DenseModel':
            self.X_train = self.X_train.reshape(self.X_train.shape[:1] + (np.prod(self.X_train.shape[1:]), ))
            self.X_test = self.X_test.reshape(self.X_test.shape[:1] + (np.prod(self.X_test.shape[1:]), ))
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.hyperparams['val_size'], random_state=0)
        self.X_train, self.X_unlabel, self.y_train, _ = train_test_split(self.X_train, self.y_train, train_size=self.hyperparams['train_size'], random_state=0)
        
        self.X_train = torch.tensor(self.X_train).cuda(self.hyperparams['gpu_id'])
        self.y_train = torch.tensor(self.y_train, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        self.X_val = torch.tensor(self.X_val).cuda(self.hyperparams['gpu_id'])
        self.y_val = torch.tensor(self.y_val, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        self.X_unlabel = torch.tensor(self.X_unlabel).cuda(self.hyperparams['gpu_id'])
        
        if self.hyperparams['model_init_fn'] == 'model.ConvModel':
            # swap axes form NHWC to NCHW
            self.X_train = self.X_train.permute(0, 3, 1, 2)
            self.X_unlabel = self.X_unlabel.permute(0, 3, 1, 2)
            self.X_val = self.X_val.permute(0, 3, 1, 2)
            
        self.X_test = torch.tensor(self.X_val).cuda(self.hyperparams['gpu_id'])
        self.y_test = torch.tensor(self.y_val).cuda(self.hyperparams['gpu_id'])
        
        # save number of classes for later use:
        self.hyperparams['n_classes'] = len(self.y_train.unique())
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            model_path = self.hyperparams['model_init_fn'].split(".")
            model_init_fn = getattr(importlib.import_module(".".join(model_path[:-1])), model_path[-1])
            
            self.model = model_init_fn(self.X_train.shape[1:], self.hyperparams['n_classes'], **self.hyperparams['model_kwargs']).cuda(self.hyperparams['gpu_id'])
            self.model_optimizer = Adam(self.model.parameters(), lr=self.hyperparams['model_lr'])
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
            
    def _get_state(self):
        state = torch.zeros(self.observation_space.shape)
        last_y_val_pred_exp = self.last_y_val_pred.exp()
        
        for i in range(self.hyperparams['n_classes']):
            mask = i == self.y_val
            state[i * self.hyperparams['n_classes']:(i+1) * self.hyperparams['n_classes']] = last_y_val_pred_exp[mask].mean(axis=0)
            
        state = state.view(-1)
        state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
        return state
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['significance_decay']) * self.reward_moving_average + self.hyperparams['significance_decay'] * reward
        if self.timestep < self.hyperparams['min_timesteps']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['significance_threshold']
            
    def step(self, action):
        # rescale action to [0, 1] range.
        action = ((action + 1) / 2).view(-1)
        range_scale = 0.5 - torch.abs(0.5 - action[0])
        action[1] = action[1] * range_scale
        self.last_action = [action[0] - action[1], action[0] + action[1]]
        
        tau1, tau2 = self.last_action[0], self.last_action[1]

        assert tau1 <= tau2
        
        y_unlabel_estimates_max, y_unlabel_estimates_argmax = torch.max(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max > tau1) & (y_unlabel_estimates_max < tau2)
        self.len_selected_samples = y_unlabel_estimates_indices.sum().double()
    
        # y_pred_binary = torch.zeros(self.last_y_unlabel_pred.shape, dtype=torch.long).scatter_(1, y_unlabel_estimates_argmax.view((-1, 1)), 1.0).cuda(self.hyperparams['gpu_id'])
        
        self.model.fit(torch.cat((self.X_train, self.X_unlabel[y_unlabel_estimates_indices]), axis=0),
                        torch.cat((self.y_train, y_unlabel_estimates_argmax[y_unlabel_estimates_indices]), axis=0),
                        optimizer=self.model_optimizer,
                        loss_fn=self.model_loss,
                        epochs=self.hyperparams['epochs_per_step'], 
                        batch_size=self.hyperparams['batch_size'], 
                        verbose=0,
                        gpu_id=self.hyperparams['gpu_id'])
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['pred_batch_size']).detach()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['pred_batch_size']).detach()
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['pred_batch_size']).detach()
        
        self.y_unlabel_estimates = (1 - self.hyperparams['y_estimated_lr']) * self.y_unlabel_estimates + self.hyperparams['y_estimated_lr'] * self.last_y_unlabel_pred.exp()
        self.y_unlabel_estimates /= self.y_unlabel_estimates.sum(axis=1).view((-1, 1))

        new_accuracy = accuracy_score(self.y_val, torch.argmax(self.last_y_val_pred, axis=1))
        
        self.last_reward = new_accuracy - self.last_val_accuracy    # -(new_mse_loss - self.last_mse_loss)    # new_accuracy - self.last_val_accuracy
        reward_scale_factor = self.hyperparams['reward_scale']
        self.last_reward *= reward_scale_factor
        
        self.last_val_accuracy = new_accuracy
        self.last_train_accuracy = accuracy_score(self.y_train, torch.argmax(self.last_y_label_pred, axis=1))
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, self.y_val)
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, self.y_train)
        
        self.last_state = self._get_state()
   
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['max_timesteps'] or not self._significant(self.last_reward) else False
            
        return self.last_state, self.last_reward, terminal, { 'val_acc': self.last_val_accuracy, 'timestep': self.timestep }
        
    def reset(self):
        self._initialize_model()
        
        self.last_action = torch.zeros(self.action_space.shape[0])
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['pred_batch_size']).detach()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['pred_batch_size']).detach()
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['pred_batch_size']).detach()
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.hyperparams['n_classes']), device=torch.device('cuda', self.hyperparams['gpu_id'])) * (1 / self.hyperparams['n_classes'])
        self.timestep = 0
        self.last_reward = 0
        # self.last_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['significance_threshold']
        
        self.last_val_accuracy = accuracy_score(self.y_val, torch.argmax(self.last_y_val_pred, axis=1))
        self.last_train_accuracy = accuracy_score(self.y_train, torch.argmax(self.last_y_label_pred, axis=1))
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, self.y_val)
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, self.y_train)
        
        self.last_state = self._get_state()

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nThresholds:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        render_string += "\nState:\n" + str(self.last_state[:-5].view((self.hyperparams['n_classes'], self.hyperparams['n_classes'])).detach().numpy())[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['worker_id']) + ".log", "w"))
        
        return render_string

    