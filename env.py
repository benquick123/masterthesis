import gym
import pickle
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from hidden_features import train_autoencoder, cluster_images
from model import MNISTModel

gym.register("SelfTeaching-v0")
gym.register("SelfTeaching-v1")

class SelfTeachingEnvV0(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v0")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, SIGNIFICANT_THRESHOLD=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, HISTORY_LEN=1, HISTORY_MEAN=False):
        super(SelfTeachingEnvV0, self).__init__()
        
        print("Initializing environment.")
        self.hyperparams = dict(locals())
                
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
        assert 'HISTORY_LEN' in self.hyperparams and self.hyperparams['HISTORY_LEN'] >= 1
        assert 'HISTORY_MEAN' in self.hyperparams and (self.hyperparams['HISTORY_MEAN'] and self.hyperparams['HISTORY_LEN'] >= 2 or not self.hyperparams['HISTORY_MEAN'])
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=((self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1) * (self.y_train.shape[1] ** 2 + 5), ))
    
    def _generate_data(self, save=True):
        print("No data exist. Generating.")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # reshape, cast and scale in one step.
        X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
        X_train, self.X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
        self.X_train, self.X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
        
        self.X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
        
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.y_val = np_utils.to_categorical(y_val)
        
        autoencoder, encoder = train_autoencoder(X_train)
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
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_unlabel = data['X_unlabel']
            self.X_unlabel_groups = data['X_unlabel_groups']
            self.X_val = data['X_val']
            self.y_val = data['y_val']
            self.X_val_groups = data['X_val_groups']
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            self.group_centers = data['group_centers']
        except FileNotFoundError:
            self._generate_data(save=True)
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            self.model = MNISTModel(self.X_train.shape[1:], self.y_train.shape[1])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.reset()
            
    def _get_state(self):
        state = []  
        y_val_argmax = np.argmax(self.y_val, axis=1)
        
        for i in range(self.y_val.shape[1]):
            mask = i == y_val_argmax
            state.append(np.mean(self.last_y_val_pred[mask], axis=0))
            
        state = np.array(state).flatten()
        state = np.append(state, [self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])

        # assert (state.shape == self.observation_space.shape) if self.hyperparams['HISTORY_LEN'] is None else (state.shape == self.observation_space.shape[1:])
        return state
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['SIGNIFICANCE_DECAY']) * self.reward_moving_average + self.hyperparams['SIGNIFICANCE_DECAY'] * reward
        if self.timestep < self.hyperparams['MIN_TIMESTEPS']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['SIGNIFICANT_THRESHOLD']
            
    def step(self, action):
        self.last_action = action.reshape(-1)
        
        tau1, tau2 = self.last_action[0], self.last_action[1]
        if tau1 >= tau2:
            tau1, tau2 = tau2, tau1

        y_unlabel_estimates_argmax = np.argmax(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_max = np.max(self.y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max > tau1) & (y_unlabel_estimates_max < tau2)
        self.len_selected_samples = np.sum(y_unlabel_estimates_indices)
        
        y_pred_binary = np.zeros_like(self.last_y_unlabel_pred)
        for i in range(len(y_unlabel_estimates_argmax)):
            y_pred_binary[i, y_unlabel_estimates_argmax[i]] = 1.0
        
        history = self.model.fit(np.concatenate([self.X_train, self.X_unlabel[y_unlabel_estimates_indices]], axis=0),
                        np.concatenate([self.y_train, y_pred_binary[y_unlabel_estimates_indices]], axis=0), 
                        epochs=self.hyperparams['EPOCHS_PER_STEP'], 
                        batch_size=self.hyperparams['BATCH_SIZE'], 
                        verbose=0)
        
        self.last_train_loss = history.history['loss'][0]
        self.last_train_accuracy = history.history['accuracy'][0]
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        
        self.y_unlabel_estimates = (1 - self.hyperparams['Y_ESTIMATED_LR']) * self.y_unlabel_estimates + self.hyperparams['Y_ESTIMATED_LR'] * self.last_y_unlabel_pred
        self.y_unlabel_estimates /= np.reshape(np.sum(self.y_unlabel_estimates, axis=1), (-1, 1))
        
        new_accuracy = accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.last_y_val_pred, axis=1))
        
        self.last_reward = (new_accuracy - self.last_val_accuracy)
        # self.last_reward *= np.exp(1 - np.abs(self.last_reward))
        # self.last_reward *= new_accuracy
        # self.last_reward *= 1.01 if self.len_selected_samples > 0 and self.last_reward > 0 else 1.0
        
        self.last_val_accuracy = new_accuracy
        self.last_val_loss = log_loss(self.y_val, self.last_y_val_pred)
        
        curr_state = self._get_state()
        self.last_state = np.append(self.last_state.reshape((self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1)), curr_state.reshape((1, -1)), axis=0)
        self.last_state = return_state = np.delete(self.last_state, 0, 0)
        if self.hyperparams['HISTORY_MEAN']:
            return_state =  np.mean(return_state, axis=0)
            
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['N_TIMESTEPS'] or not self._significant(self.last_reward) else False
            
        return np.reshape(return_state, (-1, )), self.last_reward * 10, terminal, { 'val_acc': self.last_val_accuracy, 'timestep': self.timestep }
        
    def reset(self):
        self._initialize_model()
        
        self.y_unlabel_estimates = np.ones((self.X_unlabel.shape[0], self.y_train.shape[1])) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['SIGNIFICANT_THRESHOLD']
        
        self.last_action = np.zeros(self.action_space.shape[0])
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        
        self.last_val_accuracy = accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.last_y_val_pred, axis=1))
        self.last_train_accuracy = accuracy_score(np.argmax(self.y_train, axis=1), np.argmax(self.last_y_label_pred, axis=1))
        self.last_val_loss = log_loss(self.y_val, self.last_y_val_pred)
        self.last_train_loss = log_loss(self.y_train, self.last_y_label_pred)
        
        curr_state = self._get_state()
        
        self.last_state = np.reshape(np.zeros(self.observation_space.shape), (self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1))    
        self.last_state = np.append(self.last_state.reshape((self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1)), curr_state.reshape((1, -1)), axis=0)
        self.last_state = return_state = np.delete(self.last_state, 0, axis=0)
        if self.hyperparams['HISTORY_MEAN']:
            return_state = np.mean(self.last_state, axis=0)
        
        return np.reshape(return_state, (-1, ))
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward*10)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nAction:\n" + str(['%.6f' % (element) for element in self.last_action]).replace("'", "")
        if self.hyperparams['HISTORY_LEN'] is None:
            render_string += "\nState:\n" + str(self.last_state[:-5].reshape((self.y_val.shape[1], self.y_val.shape[1])))[:-1]
            render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-5:]]).replace("'", "").replace(",", "")
        else:
            render_string += "\nState:\n" + str(self.last_state[-1][:-5].reshape((self.y_val.shape[1], self.y_val.shape[1])))[:-1]
            render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[-1][-5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + self.hyperparams['ID'] + ".log", "w"))
        
        return render_string


class SelfTeachingEnvV1(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (-10.0, 10.0)
    spec = gym.spec("SelfTeaching-v1")
    
    def __init__(self, N_CLUSTERS=50, Y_ESTIMATED_LR=0.3, N_TIMESTEPS=500, MIN_TIMESTEPS=0, BATCH_SIZE=256, PRED_BATCH_SIZE=8192, SIGNIFICANT_THRESHOLD=0.001, SIGNIFICANCE_DECAY=0.02, EPOCHS_PER_STEP=1, ID=0, HISTORY_LEN=1, HISTORY_MEAN=False):
        super(SelfTeachingEnvV1, self).__init__()
        
        print("Initializing environment.")
        self.hyperparams = dict(locals())
                
        assert 'N_CLUSTERS' in self.hyperparams and self.hyperparams['N_CLUSTERS'] > 0
        assert 'Y_ESTIMATED_LR' in self.hyperparams and 0.0 < self.hyperparams['Y_ESTIMATED_LR'] <= 1.0
        assert 'N_TIMESTEPS' in self.hyperparams and self.hyperparams['N_TIMESTEPS'] > 0
        assert 'BATCH_SIZE' in self.hyperparams
        assert 'PRED_BATCH_SIZE' in self.hyperparams
        assert 'SIGNIFICANT_THRESHOLD' in self.hyperparams
        assert 'SIGNIFICANCE_DECAY' in self.hyperparams and 0.0 <= self.hyperparams['SIGNIFICANCE_DECAY'] <= 1.0
        assert 'MIN_TIMESTEPS' in self.hyperparams
        assert 'EPOCHS_PER_STEP' in self.hyperparams
        assert 'ID' in self.hyperparams
        assert 'HISTORY_LEN' in self.hyperparams and self.hyperparams['HISTORY_LEN'] >= 1
        assert 'HISTORY_MEAN' in self.hyperparams and (self.hyperparams['HISTORY_MEAN'] and self.hyperparams['HISTORY_LEN'] >= 2 
                                                       or self.hyperparams['HISTORY_LEN'] == 1 and not self.hyperparams['HISTORY_MEAN'])
        
        self.model = None
        
        self._load_data()
        
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2 * self.hyperparams['N_CLUSTERS'], ))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.hyperparams['N_CLUSTERS'] * (self.y_train.shape[1] + 1) + 5, ))
    
    def _generate_data(self, save=True):
        print("No data exist. Generating.")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # reshape, cast and scale in one step.
        X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
        X_train, self.X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
        self.X_train, self.X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
        
        self.X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
        
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.y_val = np_utils.to_categorical(y_val)
        
        autoencoder, encoder = train_autoencoder(X_train)
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
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_unlabel = data['X_unlabel']
            self.X_unlabel_groups = data['X_unlabel_groups']
            self.X_val = data['X_val']
            self.y_val = data['y_val']
            self.X_val_groups = data['X_val_groups']
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            self.group_centers = data['group_centers']
        except FileNotFoundError:
            self._generate_data(save=True)
    
    def _initialize_model(self):
        if self.model is None:
            print("Initializing model.")
            self.model = MNISTModel(self.X_train.shape[1:], self.y_train.shape[1])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.reset()
            
    def _get_state(self):
        state = []
        
        for i in range(self.hyperparams['N_CLUSTERS']):
            mask = i == self.X_val_groups
            cluster_diff = self.last_y_val_pred[mask] - self.y_val[mask]
            cluster_diff_means = np.mean(cluster_diff, axis=0)
            cluster_diff_std = np.std(np.max(cluster_diff, axis=1))
            cluster_state = np.append(cluster_diff_means, [cluster_diff_std])
            state.append(cluster_state)
            
        state = np.array(state).flatten()
        state = np.append(state, [self.len_selected_samples / self.X_unlabel.shape[0], self.last_val_accuracy, self.last_train_accuracy, self.last_val_loss, self.last_train_loss])

        assert state.shape == self.observation_space.shape
        return state
    
    def _significant(self, reward):
        self.reward_moving_average = (1 - self.hyperparams['SIGNIFICANCE_DECAY']) * self.reward_moving_average + self.hyperparams['SIGNIFICANCE_DECAY'] * reward
        if self.timestep < self.hyperparams['MIN_TIMESTEPS']:
            return True
        else:
            return self.reward_moving_average >= self.hyperparams['SIGNIFICANT_THRESHOLD']
            
    def step(self, action):
        self.last_action = action.reshape((-1, 2))
        
        X_unlabel_selected = None
        y_unlabel_selected = None
        for i, thresholds in enumerate(self.last_action):
            tau0, tau1 = thresholds[0], thresholds[1]
            if tau0 > tau1:
                tau0, tau1 = tau1, tau0
            assert tau0 <= tau1
            
            # get only n-th group samples
            mask = i == self.X_unlabel_groups
            group_Y = self.y_unlabel_estimates[mask]
            group_X = self.X_unlabel[mask]
            
            # get samples from the same group that are between tau0 and tau1
            group_estimates_max = np.max(group_Y, axis=1)
            group_estimates_mask = (group_estimates_max > tau0) & (group_estimates_max < tau1)
            group_estimates_argmax = np.argmax(group_Y[group_estimates_mask], axis=1)
            
            group_binary = np.zeros((len(group_estimates_argmax), self.y_unlabel_estimates.shape[1]))
            for i in range(len(group_estimates_argmax)):
                group_binary[i, group_estimates_argmax[i]] = 1.0
                
            if X_unlabel_selected is None or y_unlabel_selected is None:
                X_unlabel_selected = group_X[group_estimates_mask]
                y_unlabel_selected = group_binary
            else:
                X_unlabel_selected = np.append(X_unlabel_selected, group_X[group_estimates_mask], axis=0)
                y_unlabel_selected = np.append(y_unlabel_selected, group_binary, axis=0)
                
        self.len_selected_samples = y_unlabel_selected.shape[0]
        
        history = self.model.fit(np.concatenate([self.X_train, X_unlabel_selected], axis=0),
                        np.concatenate([self.y_train, y_unlabel_selected], axis=0), 
                        epochs=self.hyperparams['EPOCHS_PER_STEP'], 
                        batch_size=self.hyperparams['BATCH_SIZE'], 
                        verbose=0)
        
        self.last_train_loss = history.history['loss'][0]
        self.last_train_accuracy = history.history['accuracy'][0]
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        
        self.y_unlabel_estimates = (1 - self.hyperparams['Y_ESTIMATED_LR']) * self.y_unlabel_estimates + self.hyperparams['Y_ESTIMATED_LR'] * self.last_y_unlabel_pred
        self.y_unlabel_estimates /= np.reshape(np.sum(self.y_unlabel_estimates, axis=1), (-1, 1))
        
        new_accuracy = accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.last_y_val_pred, axis=1))
        
        self.last_reward = (new_accuracy - self.last_val_accuracy)
        
        self.last_val_accuracy = new_accuracy
        self.last_val_loss = log_loss(self.y_val, self.last_y_val_pred)
        
        curr_state = self._get_state()
        self.last_state = np.append(self.last_state.reshape((self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1)), curr_state.reshape((1, -1)), axis=0)
        self.last_state = return_state = np.delete(self.last_state, 0, 0)
        if self.hyperparams['HISTORY_MEAN']:
            return_state =  np.mean(return_state, axis=0)
            
        self.timestep += 1
        terminal = True if self.timestep >= self.hyperparams['N_TIMESTEPS'] or not self._significant(self.last_reward) else False
            
        return np.reshape(return_state, (-1, )), self.last_reward * 10, terminal, { 'val_acc': self.last_val_accuracy, 'timestep': self.timestep }
        
    def reset(self):
        self._initialize_model()
        
        self.y_unlabel_estimates = np.ones((self.X_unlabel.shape[0], self.y_train.shape[1])) * (1 / self.y_train.shape[1])
        self.timestep = 0
        self.last_reward = 0
        self.len_selected_samples = 0
        self.reward_moving_average = self.hyperparams['SIGNIFICANT_THRESHOLD']
        
        self.last_action = np.zeros(self.action_space.shape[0])
        
        self.last_y_val_pred = self.model.predict(self.X_val, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_unlabel_pred = self.model.predict(self.X_unlabel, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        self.last_y_label_pred = self.model.predict(self.X_train, batch_size=self.hyperparams['PRED_BATCH_SIZE'])
        
        self.last_val_accuracy = accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.last_y_val_pred, axis=1))
        self.last_train_accuracy = accuracy_score(np.argmax(self.y_train, axis=1), np.argmax(self.last_y_label_pred, axis=1))
        self.last_val_loss = log_loss(self.y_val, self.last_y_val_pred)
        self.last_train_loss = log_loss(self.y_train, self.last_y_label_pred)
        
        curr_state = self._get_state()
        
        self.last_state = np.reshape(np.zeros(self.observation_space.shape), (self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1))    
        self.last_state = np.append(self.last_state.reshape((self.hyperparams['HISTORY_LEN'] if not self.hyperparams['HISTORY_MEAN'] else 1, -1)), curr_state.reshape((1, -1)), axis=0)
        self.last_state = return_state = np.delete(self.last_state, 0, axis=0)
        if self.hyperparams['HISTORY_MEAN']:
            return_state = np.mean(self.last_state, axis=0)
        
        return np.reshape(return_state, (-1, ))
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f" % (self.timestep, self.last_reward*10)
        render_string += "\nLOSS: %.3f - TRAIN_ACC: %.3f - VAL_ACC: %.3f" % (self.last_train_loss, self.last_train_accuracy, self.last_val_accuracy)
        render_string += "\nSignificance level: %.3f" % (self.reward_moving_average)
        render_string += "\nNum. selected samples: %d" % (self.len_selected_samples)
        
        render_string += "\n\nAction:\n" + str(np.mean(self.last_action)) + "(+/- " + str(np.std(self.last_action)) + ")"
        render_string += "\nState:\n" + str(self.last_state[0, :-5].reshape((self.hyperparams['N_CLUSTERS'], self.y_val.shape[1] + 1)))[:-1]
        render_string += "\n " + str(['%.2f' % (element) for element in self.last_state[0, -5:]]).replace("'", "").replace(",", "")
        
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp_" + self.hyperparams['ID'] + ".log", "w"))
        
        return render_string