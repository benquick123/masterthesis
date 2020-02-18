import gym
import pickle
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from datetime import datetime

import numpy as np # only for dataset construction

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss
from torch import Tensor

from hidden_features import train_autoencoder, cluster_images
from model import MNISTModel

gym.register("SelfTeaching-v0")
gym.register("SelfTeaching-v1")
gym.register("SelfTeaching-v2")

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum().double() / y_true.shape[0]


class SelfTeachingEnvV0(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v0")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, SIGNIFICANT_THRESHOLD=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, REWARD_SCALE=10):
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
        assert 'SIGNIFICANT_THRESHOLD' in self.hyperparams
        assert 'SIGNIFICANCE_DECAY' in self.hyperparams and 0.0 <= self.hyperparams['SIGNIFICANCE_DECAY'] <= 1.0
        assert 'MIN_TIMESTEPS' in self.hyperparams
        assert 'EPOCHS_PER_STEP' in self.hyperparams
        assert 'ID' in self.hyperparams
        assert 'REWARD_SCALE' in self.hyperparams and self.hyperparams['REWARD_SCALE'] > 0.0
        
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
                'y_val': self.y_val,
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
            self.X_unlabel_groups = torch.Tensor(data['X_unlabel_groups']).cuda(self.hyperparams['GPU_ID'])
            self.X_val = torch.Tensor(data['X_val']).cuda(self.hyperparams['GPU_ID'])
            self.y_val = torch.Tensor(data['y_val']).cuda(self.hyperparams['GPU_ID'])
            self.X_val_groups = torch.Tensor(data['X_val_groups']).cuda(self.hyperparams['GPU_ID'])
            self.X_test = torch.Tensor(data['X_test']).cuda(self.hyperparams['GPU_ID'])
            self.y_test = torch.Tensor(data['y_test']).cuda(self.hyperparams['GPU_ID'])
            self.group_centers = torch.Tensor(data['group_centers']).cuda(self.hyperparams['GPU_ID'])
        except FileNotFoundError:
            self._generate_data(save=True)
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            self.model = MNISTModel(self.X_train.shape[1], self.y_train.shape[1]).cuda(self.hyperparams['GPU_ID'])
            self.model.reset()
            self.model_optimizer = Adam(self.model.parameters(), lr=0.001)
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
            
    def _get_state(self):
        state = torch.zeros(self.observation_space.shape).cuda(self.hyperparams['GPU_ID'])
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
            return self.reward_moving_average >= self.hyperparams['SIGNIFICANT_THRESHOLD']
            
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
    
        y_pred_binary = torch.zeros(self.last_y_unlabel_pred.shape).cuda().scatter_(1, y_unlabel_estimates_argmax.view((-1, 1)), 1.0)
        
        self.model.fit(torch.cat((self.X_train, self.X_unlabel[y_unlabel_estimates_indices]), axis=0),
                        torch.cat((self.y_train, y_pred_binary[y_unlabel_estimates_indices]), axis=0),
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
        
        self.last_action = torch.zeros(self.action_space.shape[0]).cuda(self.hyperparams['GPU_ID'])
        
        self.last_y_val_pred = self.model.predict(self.X_val).detach()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel).detach()
        self.last_y_label_pred = self.model.predict(self.X_train).detach()
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.y_train.shape[1])).cuda(self.hyperparams['GPU_ID']) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        # self.last_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['SIGNIFICANT_THRESHOLD']
        
        self.last_val_accuracy = accuracy_score(torch.argmax(self.y_val, axis=1), torch.argmax(self.last_y_val_pred, axis=1)).detach()
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1)).detach()
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val, axis=1)).detach()
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1)).detach()
        
        self.last_state = self._get_state().detach()

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nThresholds:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        render_string += "\nState:\n" + str(self.last_state[:-5].view((self.y_val.shape[1], self.y_val.shape[1])).cpu().detach().numpy())[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['ID']) + ".log", "w"))
        
        return render_string


class SelfTeachingEnvV1(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v1")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, SIGNIFICANT_THRESHOLD=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, REWARD_SCALE=10):
        super(SelfTeachingEnvV1, self).__init__()
        
        print("Initializing environment.")
        self.hyperparams = dict(locals())
        del self.hyperparams['self']
        del self.hyperparams['__class__']
                
        assert 'N_CLUSTERS' in self.hyperparams and self.hyperparams['N_CLUSTERS'] > 0
        assert 'Y_ESTIMATED_LR' in self.hyperparams and 0.0 < self.hyperparams['Y_ESTIMATED_LR'] < 1.0
        assert 'N_TIMESTEPS' in self.hyperparams and self.hyperparams['N_TIMESTEPS'] > 0
        assert 'BATCH_SIZE' in self.hyperparams
        assert 'PRED_BATCH_SIZE' in self.hyperparams
        assert 'SIGNIFICANT_THRESHOLD' in self.hyperparams
        assert 'SIGNIFICANCE_DECAY' in self.hyperparams and 0.0 <= self.hyperparams['SIGNIFICANCE_DECAY'] <= 1.0
        assert 'MIN_TIMESTEPS' in self.hyperparams
        assert 'EPOCHS_PER_STEP' in self.hyperparams
        assert 'ID' in self.hyperparams
        assert 'REWARD_SCALE' in self.hyperparams and self.hyperparams['REWARD_SCALE'] > 0.0
        
        self.hyperparams['GPU_ID'] = self.hyperparams['ID'] % torch.cuda.device_count()
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 * self.hyperparams['N_CLUSTERS'], ))
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.y_train.shape[1] * self.hyperparams['N_CLUSTERS'] + 5, )))
    
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
                'y_val': self.y_val,
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
            self.model = MNISTModel(self.X_train.shape[1], self.y_train.shape[1]).cuda(self.hyperparams['GPU_ID'])
            self.model.reset()
            self.model_optimizer = Adam(self.model.parameters(), lr=0.001)
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
            
    def _get_state(self):
        state = torch.zeros(self.observation_space.shape).cuda(self.hyperparams['GPU_ID'])
        last_y_val_pred_exp = self.last_y_val_pred.exp()
        
        for i in range(self.hyperparams['N_CLUSTERS']):
            mask = i == self.X_val_groups
            state[i * self.y_val.shape[1]:(i+1) * self.y_val.shape[1]] = (last_y_val_pred_exp[mask] - self.y_val[mask]).mean(axis=0)
        
        state = state.view(-1)
        state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
        return state
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['SIGNIFICANCE_DECAY']) * self.reward_moving_average + self.hyperparams['SIGNIFICANCE_DECAY'] * reward
        if self.timestep < self.hyperparams['MIN_TIMESTEPS']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['SIGNIFICANT_THRESHOLD']
            
    def step(self, action):
        # rescale action from [-1, 1] to [0, 1] range.
        action = ((action + 1) / 2).view((-1, 2))
        range_scale = 0.5 - torch.abs(0.5 - action[:, 0])
        action[:, 1] = action[:, 1] * range_scale
        self.last_action = torch.zeros(action.shape).cuda(self.hyperparams['GPU_ID'])
        self.last_action[:, 0] = action[:, 0] - action[:, 1]
        self.last_action[:, 1] = action[:, 0] + action[:, 1]

        assert torch.all(self.last_action[:, 0] <= self.last_action[:, 1])

        tau = torch.index_select(self.last_action, 0, self.X_unlabel_groups)
        
        y_unlabel_estimates_max, y_unlabel_estimates_argmax = torch.max(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max > tau[:, 0]) & (y_unlabel_estimates_max < tau[:, 1])
        self.len_selected_samples = y_unlabel_estimates_indices.sum().double()
        
        y_pred_binary = torch.zeros(self.last_y_unlabel_pred.shape).cuda().scatter_(1, y_unlabel_estimates_argmax.view((-1, 1)), 1.0)
        
        self.model.fit(torch.cat((self.X_train, self.X_unlabel[y_unlabel_estimates_indices]), axis=0),
                        torch.cat((self.y_train, y_pred_binary[y_unlabel_estimates_indices]), axis=0),
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

        new_accuracy = accuracy_score(torch.argmax(self.y_val, axis=1), torch.argmax(self.last_y_val_pred, axis=1))
        # new_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        
        self.last_reward = new_accuracy - self.last_val_accuracy    # -(new_mse_loss - self.last_mse_loss)
        reward_scale_factor = self.hyperparams['REWARD_SCALE']
        self.last_reward *= reward_scale_factor
        # self.last_mse_loss = new_mse_loss
        
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
        
        self.last_action = torch.zeros(self.action_space.shape[0]).cuda(self.hyperparams['GPU_ID']).view((-1, 2))
        
        self.last_y_val_pred = self.model.predict(self.X_val).detach()
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel).detach()
        self.last_y_label_pred = self.model.predict(self.X_train).detach()
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.y_train.shape[1])).cuda(self.hyperparams['GPU_ID']) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        # self.last_mse_loss = self.mse_loss(self.last_y_val_pred.exp(), self.y_val)
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['SIGNIFICANT_THRESHOLD']
        
        self.last_val_accuracy = accuracy_score(torch.argmax(self.y_val, axis=1), torch.argmax(self.last_y_val_pred, axis=1)).detach()
        self.last_train_accuracy = accuracy_score(torch.argmax(self.y_train, axis=1), torch.argmax(self.last_y_label_pred, axis=1)).detach()
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, torch.argmax(self.y_val, axis=1)).detach()
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, torch.argmax(self.y_train, axis=1)).detach()
        
        self.last_state = self._get_state().detach()

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nState & action:\n"
        for state_part, action_part in zip(self.last_state[:-5].view((self.hyperparams['N_CLUSTERS'], self.y_val.shape[1])).cpu().detach().numpy(), self.last_action.cpu().detach().numpy()):
            render_string += str(state_part) + "    \t" + str(action_part) + "\n"
        render_string += str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['ID']) + ".log", "w"))
        
        return render_string

