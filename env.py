import gym
import pickle
import json
import os
import importlib
import inspect
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from datetime import datetime

from ignite.metrics import ConfusionMatrix

import numpy as np # only for dataset construction

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss, MSELoss
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from utils import Logger

gym.register("SelfTeachingBase-v0")


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum().double() / y_true.shape[0]


class SelfTeachingBaseEnv(gym.Env):
    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeachingBase-v0")
    
    def __init__(self, dataset, config_path, logger=Logger(), override_hyperparams={}):
        super(SelfTeachingBaseEnv, self).__init__()
        
        self.logger = logger
        
        default_hyperparams = json.load(open(os.path.join(config_path, "defaults.json")))
        experiment_hyperparams = json.load(open(os.path.join(config_path, dataset.lower() + ".json"), "r"))
        # take care for correct hyperparameters initialization
        self.hyperparams = dict(default_hyperparams)
        self.hyperparams.update(experiment_hyperparams)
        self.hyperparams.update(override_hyperparams)
        
        # weird lambda trick that freezes the 'unlabel_alpha' parameter.
        self.hyperparams['unlabel_alpha'] = (lambda step, alpha=self.hyperparams['unlabel_alpha'] : alpha) if self.hyperparams['unlabel_alpha'] is not None else self._get_alpha
        self.hyperparams['gpu_id'] = self.hyperparams['worker_id'] % torch.cuda.device_count()
        
        self.logger.print("Initializing environment:", self.hyperparams['worker_id'])
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=((self.hyperparams['output_state_dim'] ** 2 + 5, )))
        
        self.is_testing = False
        self.known_labels = False
    
    def _load_data(self):
        self.logger.print("Loading data:", self.hyperparams['name'])
        module_path = self.hyperparams['loader_fn'].split(".")
        dataset_loader_fn = getattr(importlib.import_module(".".join(module_path[:-1])), module_path[-1])
        
        assert "train" not in self.hyperparams['loader_kwargs']
        # implementation assumes a interface otherwise found in torchvision.datasets.
        loader_fn_params = inspect.signature(dataset_loader_fn).parameters
        if 'train' in loader_fn_params:
            train = dataset_loader_fn(train=True, **self.hyperparams['loader_kwargs'])
            test = dataset_loader_fn(train=False, **self.hyperparams['loader_kwargs'])
        elif 'split' in loader_fn_params:
            train = dataset_loader_fn(split="train", **self.hyperparams['loader_kwargs'])
            test = dataset_loader_fn(split="test", **self.hyperparams['loader_kwargs'])
        else:
            self.logger.print("Function doesn't accept 'train' or 'split' parameter.")
            raise AttributeError
        
        # ensure train and test are numpy arrays by first converting them to Tensor and then back to np.array
        self.X_train = torch.tensor(getattr(train, self.hyperparams['dataset_data'])).numpy()
        self.y_train = torch.tensor(getattr(train, self.hyperparams['dataset_labels'])).numpy()
        self.X_test = torch.tensor(getattr(test, self.hyperparams['dataset_data'])).numpy()
        self.y_test = torch.tensor(getattr(test, self.hyperparams['dataset_labels'])).numpy()
        
        if isinstance(self.hyperparams['train_size'], float):
            self.hyperparams['train_size'] = int(len(X_train) * self.hyperparams['train_size'])
        if isinstance(self.hyperparams['val_size'], float):
            self.hyperparams['val_size'] = int(len(X_train) * self.hyperparams['val_size'])
        
        if self.hyperparams['model_init_fn'] == 'model.DenseModel':
            self.X_train = self.X_train.astype('float32')
            self.X_test = self.X_test.astype('float32')
            self.X_train = self.X_train.reshape(self.X_train.shape[:1] + (np.prod(self.X_train.shape[1:]), ))
            self.X_test = self.X_test.reshape(self.X_test.shape[:1] + (np.prod(self.X_test.shape[1:]), ))
            
        if self.hyperparams['model_init_fn'] == 'model.ConvModel':
            self.X_train = self.X_train.astype('float32')
            self.X_test = self.X_test.astype('float32')
            
            if len(self.X_train.shape) == 3:
                self.X_train = np.expand_dims(self.X_train, -1)
            if len(self.X_test.shape) == 3:
                self.X_test = np.expand_dims(self.X_test, -1)
                
            # swap axes from NHWC to NCHW, if necessary.
            if self.X_train.shape[1] not in {1, 3}:
                self.X_train = np.transpose(self.X_train, (0, 3, 1, 2))
            if self.X_test.shape[1] not in {1, 3}:
                self.X_test = np.transpose(self.X_test, (0, 3, 1, 2))
            
        if np.max(self.X_train) > 1.0 and self.hyperparams['model_init_fn'] in {'model.ConvModel', 'model.DenseModel'}:
            self.X_train = self.X_train / 255
            self.X_test = self.X_test / 255
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.hyperparams['val_size'], random_state=0)
        self.X_train, self.X_unlabel, self.y_train, self.y_unlabel = train_test_split(self.X_train, self.y_train, train_size=self.hyperparams['train_size'], random_state=0)
        
        self.X_train = torch.tensor(self.X_train).cuda(self.hyperparams['gpu_id'])
        self.y_train = torch.tensor(self.y_train, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        
        self.X_val = torch.tensor(self.X_val).cuda(self.hyperparams['gpu_id'])
        self.y_val = torch.tensor(self.y_val, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        
        self.X_unlabel = torch.tensor(self.X_unlabel).cuda(self.hyperparams['gpu_id'])
        self.y_unlabel = torch.tensor(self.y_unlabel, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        
        self.X_test = torch.tensor(self.X_val).cuda(self.hyperparams['gpu_id'])
        self.y_test = torch.tensor(self.y_val, dtype=torch.long).cuda(self.hyperparams['gpu_id'])
        
        # save number of classes for later use:
        self.hyperparams['n_classes'] = len(self.y_train.unique())
        if self.hyperparams['output_state_dim'] is None:
            self.hyperparams['output_state_dim'] = self.hyperparams['n_classes']
           
    def _initialize_model(self):
        if self.model is None:
            self.logger.print("Initializing model.")
            model_path = self.hyperparams['model_init_fn'].split(".")
            model_init_fn = getattr(importlib.import_module(".".join(model_path[:-1])), model_path[-1])
            
            self.model = model_init_fn(self.X_train.shape[1:], self.hyperparams['n_classes'], **self.hyperparams['model_kwargs']).cuda(self.hyperparams['gpu_id'])
            self.model_optimizer = Adam(self.model.parameters(), lr=self.hyperparams['model_lr'])
            self.model_loss = CrossEntropyLoss()
            self.nll_loss = NLLLoss()
            self.mse_loss = MSELoss()
        
        self.model.reset()
     
    def _transform_state(self, state):
        probs = state[:-5]
        probs = probs.view((int(probs.shape[0] ** (1/2)), -1))
        
        diff = self.hyperparams['output_state_dim'] - probs.shape[0]
        probs = F.pad(probs, pad=(0, diff, 0, diff))

        state = torch.cat([probs.reshape(-1), state[-5:]])
        
        return state
            
    def _get_state(self):
        state = torch.zeros((self.hyperparams['n_classes'] ** 2 + 5))
        last_y_val_pred_exp = self.last_y_val_pred.exp()
        
        for i in range(self.hyperparams['n_classes']):
            mask = i == self.y_val
            state[i * self.hyperparams['n_classes']:(i+1) * self.hyperparams['n_classes']] = last_y_val_pred_exp[mask].mean(axis=0)

        state[-5:] = torch.Tensor([self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])
        return self._transform_state(state)
    
    def _significant(self, reward):
        self.reward_history[self.timestep % self.hyperparams['reward_history_length']] = reward
        if self.timestep < self.hyperparams['min_timesteps']:
            return True
        else:
            return torch.mean(self.reward_history) >= self.hyperparams['reward_history_threshold']
            
    def _get_alpha(self, step):
        if step <= 100:
            return 0.0
        elif 100 < step < 300:
            return 3.0 * (step - 100) / (300 - 100)
        else:
            return 3.0
    
    def step(self, action):
        # rescale action to [0, 1] range.
        action = ((action + 1) / 2).view(-1)
        range_scale = 0.5 - torch.abs(0.5 - action[0])
        action[1] = action[1] * range_scale
        self.last_action = [action[0] - action[1], action[0] + action[1]]
        
        tau1, tau2 = self.last_action[0], self.last_action[1]
        assert tau1 <= tau2
        
        y_unlabel_estimates_max, y_unlabel_estimates_argmax = torch.max(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max >= tau1) & (y_unlabel_estimates_max <= tau2)
        self.len_selected_samples = y_unlabel_estimates_indices.sum().double()
        
        self.model.fit_semi(self.X_train, self.y_train, 
                            self.X_unlabel[y_unlabel_estimates_indices], 
                            y_unlabel_estimates_argmax[y_unlabel_estimates_indices] if not self.known_labels else self.y_unlabel,
                            optimizer=self.model_optimizer,
                            loss_fn=self.model_loss,
                            epochs=self.hyperparams['epochs_per_step'], 
                            batch_size=self.hyperparams['batch_size'], 
                            verbose=0,
                            gpu_id=self.hyperparams['gpu_id'],
                            alpha=self.hyperparams['unlabel_alpha'](self.timestep))
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        
        self.y_unlabel_estimates = (1 - self.hyperparams['y_estimated_lr']) * self.y_unlabel_estimates + self.hyperparams['y_estimated_lr'] * self.last_y_unlabel_pred.exp()
        self.y_unlabel_estimates /= self.y_unlabel_estimates.sum(axis=1).view((-1, 1))

        new_accuracy = accuracy_score(self.y_val, torch.argmax(self.last_y_val_pred, axis=1))
        
        self.last_reward = new_accuracy - self.last_val_accuracy
        reward_scale_factor = self.hyperparams['reward_scale']
        self.last_reward *= reward_scale_factor
        
        self.last_val_accuracy = new_accuracy
        self.last_train_accuracy = accuracy_score(self.y_train, torch.argmax(self.last_y_label_pred, axis=1))
        
        if self.is_testing:
            self.last_y_test_pred = self.model.predict(self.X_test, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
            self.last_test_accuracy = accuracy_score(self.y_test, torch.argmax(self.last_y_test_pred, axis=1))
        
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, self.y_val)
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, self.y_train)
        
        self.last_state = self._get_state()
   
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['max_timesteps'] or not self._significant(self.last_reward) else False
        info = {'acc': self.last_val_accuracy if not self.is_testing else self.last_test_accuracy, 
                'timestep': self.timestep, 
                'num_samples': self.len_selected_samples, 
                'true_action': self.last_action}
        
        return self.last_state, self.last_reward, terminal, info
    
    def reset(self):
        self._initialize_model()
        
        self.last_action = torch.zeros(self.action_space.shape[0])
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
        
        self.y_unlabel_estimates = torch.ones((self.X_unlabel.shape[0], self.hyperparams['n_classes'])).cuda(self.hyperparams['gpu_id']) / self.hyperparams['n_classes']
        self.timestep = 0
        self.last_reward = 0
        self.len_selected_samples = 0
        self.reward_history = torch.ones((self.hyperparams['reward_history_length'], ))
        
        self.last_val_accuracy = accuracy_score(self.y_val, torch.argmax(self.last_y_val_pred, axis=1))
        self.last_train_accuracy = accuracy_score(self.y_train, torch.argmax(self.last_y_label_pred, axis=1))
        
        if self.is_testing:
            self.last_y_test_pred = self.model.predict(self.X_test, batch_size=self.hyperparams['pred_batch_size'], gpu_id=self.hyperparams['gpu_id'])
            self.last_test_accuracy = accuracy_score(self.y_test, torch.argmax(self.last_y_test_pred, axis=1))
        
        self.last_val_loss = self.nll_loss(self.last_y_val_pred, self.y_val)
        self.last_train_loss = self.nll_loss(self.last_y_label_pred, self.y_train)
        
        self.last_state = self._get_state()

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += " - TEST_ACC: %.3f" % (self.last_test_accuracy) if self.is_testing else ""
        render_string += "\nSignificance level: %.3f" % (torch.mean(self.reward_history).numpy())
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nThresholds:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        render_string += "\nState:\n" + str(self.last_state[:-5].view((self.hyperparams['output_state_dim'], self.hyperparams['output_state_dim'])).detach().numpy())[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")

        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + str(self.hyperparams['worker_id']) + ".log", "w"))
        
        return render_string

    def close(self):
        super(SelfTeachingBaseEnv, self).close()
        
        self.X_train = self.X_train.cpu().detach()
        self.y_train = self.y_train.cpu().detach()
        self.X_val = self.X_val.cpu().detach()
        self.y_val = self.y_val.cpu().detach()
        self.X_unlabel = self.X_unlabel.cpu().detach()
        self.y_unlabel = self.y_unlabel.cpu().detach()
        self.X_test = self.X_test.cpu().detach()
        self.y_test = self.y_test.cpu().detach()
        
        # self.y_unlabel_estimates = self.y_unlabel_estimates.cpu().detach()
        
        del self.X_train
        del self.y_train
        del self.X_val
        del self.y_val
        del self.X_unlabel
        del self.y_unlabel
        del self.X_test
        del self.y_test
        
        # del self.y_unlabel_estimates
        
        torch.cuda.empty_cache()


