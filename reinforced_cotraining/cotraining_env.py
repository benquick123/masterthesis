import gym
from gym import spaces
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import Adam

from preprocess import load_data
from model import SimpleModel

gym.register("Cotraining-v0")

class CotrainEnv(gym.Env):

    metadata = {'render.modes': ["ansi"]}
    reward_range = (0.0, 1.0)
    spec = gym.spec("Cotraining-v0")
    
    def __init__(self, 
                 N_CLUSTERS=80,
                 N_EPISODES_WARMUP=5,
                 N_TIMESTEPS=30,
                 discrete=True):
        self.hyperparams = {
            "N_CLUSTERS": N_CLUSTERS,
            "N_EPISODES_WARMUP": N_EPISODES_WARMUP,
            "N_TIMESTEPS": N_TIMESTEPS
        }
        
        self.X1_train, self.X2_train, self.y_train, self.X1_unlabel, self.X2_unlabel, self.X1_test, self.X2_test, self.y_test = load_data(encode=False)
        self.X1_val, self.X2_val, self.y_val = self.X1_test[:len(self.X1_test) // 2], self.X2_test[:len(self.X2_test) // 2], self.y_test[:len(self.X2_test) // 2]
        self.X1_test, self.X2_test, self.y_test = self.X1_test[len(self.X1_test) // 2:], self.X2_test[len(self.X2_test) // 2:], self.y_test[len(self.X2_test) // 2:]
        
        self.groups = self._get_groups(np.concatenate([self.X1_unlabel, self.X2_unlabel], axis=1))
        
        if discrete:
            self.action_space = spaces.Discrete(N_CLUSTERS + 1)
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N_CLUSTERS + 1, ))
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(N_CLUSTERS * 2 * 2, ))
        
        self.clf1, self.clf2 = None, None
        
    def _initialize_model(self):
        # get input shapes
        # X1_input_shape = self.X1_train.shape[1:]
        # X2_input_shape = self.X2_train.shape[1:]
        # output_shape = self.y_train.shape[1]
        
        ################################### INITIALIZE MODELS ###################################
        if self.clf1 is None or self.clf2 is None:
            self.clf1 = RandomForestClassifier(n_estimators=100)
            self.clf2 = RandomForestClassifier(n_estimators=100)
            # self.clf1 = SimpleModel(X1_input_shape, output_shape, layer_dims=[self.X1_train.shape[1] // 100, self.X1_train.shape[1] // 200])
            # self.clf1.compile(optimizer=Adam(lr=0.0005), loss="categorical_crossentropy")
            
            # self.clf2 = SimpleModel(X2_input_shape, output_shape, layer_dims=[self.X2_train.shape[1] // 5, self.X2_train.shape[1] // 10])
            # self.clf2.compile(optimizer=Adam(lr=0.0005), loss="categorical_crossentropy")
        else:
            self.clf1 = RandomForestClassifier(n_estimators=100)
            self.cfl2 = RandomForestClassifier(n_estimators=100)
            # self.clf1.reset()
            # self.clf2.reset()
        
    def _get_groups(self, X):
        """def jaccard(x1, x2):
            x1[x1 != 0.0] = 1.0
            x2[x2 != 0.0] = 1.0
            x1 = np.array(x1, dtype="bool")
            x2 = np.array(x2, dtype="bool")
            return np.sum(np.logical_and(x1, x2)) / np.sum(np.logical_or(x1, x2))"""
        
        clustering = AgglomerativeClustering(n_clusters=self.hyperparams["N_CLUSTERS"])
        return clustering.fit_predict(X)
    
    def _get_state(self, y1_pred, y2_pred):
        cluster_index = np.arange(self.hyperparams["N_CLUSTERS"])
        new_state = np.zeros(self.hyperparams["N_CLUSTERS"] * 2 * 2)
        for cluster in cluster_index:
            mask = self.groups == cluster
            y1_mean = np.mean(y1_pred[mask], axis=0)
            y2_mean = np.mean(y2_pred[mask], axis=0)
            y_mean = np.append(y1_mean, y2_mean)
            
            new_state[cluster*4:cluster*4 + len(y_mean)] = y_mean
            
        return new_state
    
    def _get_reward(self, new_accuracy):
        def r(clf_index):
            return new_accuracy[clf_index] - self.last_accuracy[clf_index]
        
        r_0 = r(0)
        r_1 = r(1)
        return 0.0 if r_0 < 0.0 or r_1 < 0.0 else r_0 * r_1
    
    def step(self, action):
        # Choose the action a_t = max_a Q(s_t; a)
        # action = self.hyperparams["N_CLUSTERS"]
        self.last_action = action
        mask = self.groups == action
        self.last_mask_sum = np.sum(mask)
        
        # Use C_1 to label the subset U_at
        y1_pred = self.last_y_pred[0][mask]
        y1_pred_binary = np.argmax(y1_pred, axis=1)     # np.round(y1_pred)
        # Update C_2 with pseudo-labeled U_at, L
        # self.clf2.fit(np.append(self.X2_train, self.X2_unlabel[mask], axis=0), np.append(self.y_train, y1_pred_binary, axis=0), epochs=1, verbose=0)
        self.clf2.fit(np.append(self.X2_train, self.X2_unlabel[mask], axis=0), np.append(self.y_train, y1_pred_binary, axis=0))
        
        # Use C_2 to label the subset U_at
        y2_pred = self.last_y_pred[1][mask]
        y2_pred_binary = np.argmax(y2_pred, axis=1)     # np.round(y2_pred)
        # Update C_1 with pseudo-labeled U_at, L
        # self.clf1.fit(np.append(self.X1_train, self.X1_unlabel[mask], axis=0), np.append(self.y_train, y2_pred_binary, axis=0), epochs=1, verbose=0)
        self.clf1.fit(np.append(self.X1_train, self.X1_unlabel[mask], axis=0), np.append(self.y_train, y2_pred_binary, axis=0))
        
        # Compute the reward r_t based on L'
        # new_accuracy = [accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.clf1.predict_proba(self.X1_val), axis=1)), 
        #                 accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(self.clf2.predict_proba(self.X2_val), axis=1))]
        new_accuracy = [accuracy_score(self.y_val, np.argmax(self.clf1.predict_proba(self.X1_val), axis=1)), 
                        accuracy_score(self.y_val, np.argmax(self.clf2.predict_proba(self.X2_val), axis=1))]
        self.last_reward = self._get_reward(new_accuracy)
        self.last_accuracy = new_accuracy
        
        # Compute the state representation s_(t+1)
        self.last_y_pred = [self.clf1.predict_proba(self.X1_unlabel), self.clf2.    predict_proba(self.X2_unlabel)]
        self.last_state = self._get_state(*self.last_y_pred)
        
        terminal = False if self.timestep < self.hyperparams["N_TIMESTEPS"] else True
        info = {"val_acc": self.last_accuracy}
        self.timestep += 1
        
        return self.last_state, self.last_reward, terminal, info
    
    def reset(self):
        self._initialize_model()
        
        if self.hyperparams["N_EPISODES_WARMUP"]:
            # self.clf1.fit(self.X1_train, self.y_train, epochs=self.hyperparams["N_EPISODES_WARMUP"], verbose=0)
            # self.clf2.fit(self.X2_train, self.y_train, epochs=self.hyperparams["N_EPISODES_WARMUP"], verbose=0)
            self.clf1.fit(self.X1_train, self.y_train)
            self.clf2.fit(self.X2_train, self.y_train)
            
        self.timestep = 0
        
        y1_pred = self.clf1.predict_proba(self.X1_val)
        y2_pred = self.clf2.predict_proba(self.X2_val)
        # self.last_accuracy = [accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(y1_pred, axis=1)), accuracy_score(np.argmax(self.y_val, axis=1), np.argmax(y2_pred, axis=1))]
        self.last_accuracy = [accuracy_score(self.y_val, np.argmax(y1_pred, axis=1)), accuracy_score(self.y_val, np.argmax(y2_pred, axis=1))]
        
        self.last_y_pred = [self.clf1.predict_proba(self.X1_unlabel), self.clf2.predict_proba(self.X2_unlabel)]
        self.last_state = self._get_state(*self.last_y_pred)
        
        self.last_action = -1
        self.last_mask_sum = 0
        self.last_reward = self._get_reward(self.last_accuracy)

        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - REWARD: %.3f - VAL_ACC: [%.3f, %.3f]" % (self.timestep, self.last_reward, self.last_accuracy[0], self.last_accuracy[1])
        
        render_string += "\n\naction: %d" % (self.last_action)
        if self.last_action == self.hyperparams["N_CLUSTERS"]:
            render_string += " == NO ACTION"
            
        render_string += " - mask_sum: %d" % (self.last_mask_sum)
        
        render_string += "\n\nstate (min, max): %.3f (%d), %.3f (%d)" % (
            np.min(self.last_state), np.argmin(self.last_state), np.max(self.last_state), np.argmax(self.last_state)
        )
        render_string += "\n" + str(self.last_state)
        
        # print("Mode should be ANSI. Printing in render():", render_string, end="\r")
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp.log", "w"))
        
        return render_string
    