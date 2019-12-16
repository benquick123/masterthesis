import os

import gym
from gym import spaces
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras as k
import time

from model import SimpleConvNet
from preprocess import SVHN, preprocess
from utils import data_generator, get_cluster_centers

gym.register("Curriculum-v0")

class CurriculumEnv(gym.Env):
    
    metadata = {'render.modes': ["ansi"]}
    reward_range = (-1.0, 1.0)
    spec = gym.spec("Curriculum-v0")
    
    def __init__(self, 
                 N_CLASSES=None, 
                 N_TIMESTEPS = 50, 
                 N_LABEL = 750, N_UNLABEL = 5000, N_VALID = 250, 
                 BATCH_SIZE = 100, 
                 N_CLUSTERS=50, CLUSTER_DELTA=None, 
                 SVHN_FILEPATH="/opt/workspace/host_storage_hdd/",
                 n_episodes_warmup=0,
                 discrete=False):
        
        self.hyperparams = {
            "N_TIMESTEPS": N_TIMESTEPS,
            "N_LABEL": N_LABEL,
            "N_UNLABEL": N_UNLABEL,
            "N_VALID": N_VALID,
            "BATCH_SIZE": BATCH_SIZE,
            "N_CLUSTERS": N_CLUSTERS,
            "SVHN_FILEPATH": SVHN_FILEPATH,
            "N_EPISODES_WARMUP": n_episodes_warmup
        }
        
        assert self.hyperparams["N_TIMESTEPS"] > 0
        assert self.hyperparams["N_LABEL"] > 0
        assert self.hyperparams["N_VALID"] > 0
        assert self.hyperparams["BATCH_SIZE"] > 0
        assert self.hyperparams["N_CLUSTERS"] > 0
        assert os.path.exists(self.hyperparams["SVHN_FILEPATH"])
        
        self._load_data(N_CLASSES, CLUSTER_DELTA)
        
        assert self.hyperparams["N_CLASSES"] > 0
        # so we avoid the error in later training
        assert self.hyperparams["N_VALID"] % self.hyperparams["N_CLASSES"] == 0
        assert self.hyperparams["N_UNLABEL"] % self.hyperparams["N_CLASSES"] == 0
        
        # self.reset()
        
        if discrete:
            self.action_space = spaces.MultiBinary(N_CLUSTERS)
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N_CLUSTERS, ))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(N_CLUSTERS, ))
        
    def _load_data(self, N_CLASSES, CLUSTER_DELTA):
        ################################## DATA LOADING ####################################
        # load data
        # IMPORTANT: watch out when using test data. Number of classes is not necessarily matched.
        (x_train, y_train), _ = SVHN.load_data(path=self.hyperparams["SVHN_FILEPATH"])
        (self.x_train_labeled, self.y_train_labeled), (self.x_train_unlabeled, self.y_train_unlabeled), (self.x_valid, self.y_valid) = preprocess(x_train, y_train, self.hyperparams["N_LABEL"], self.hyperparams["N_UNLABEL"], self.hyperparams["N_VALID"], n_classes=N_CLASSES)
        
        # select appropriate number of classes and set related hyperparams
        self.hyperparams["N_CLASSES"] = self.y_train_labeled.shape[1]
        self.hyperparams["CLUSTERS_DELTA"] = CLUSTER_DELTA if CLUSTER_DELTA is not None else 1 / self.hyperparams["N_CLASSES"]
        
        # initialize tensorflow dataset
        labeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[self.x_train_labeled, self.y_train_labeled]).shuffle(
            self.hyperparams["N_LABEL"] * 2).batch(
            self.hyperparams["N_LABEL"] * self.hyperparams["BATCH_SIZE"] // (self.hyperparams["N_LABEL"] + self.hyperparams["N_UNLABEL"]))
        unlabeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[self.x_train_unlabeled, self.y_train_unlabeled]).shuffle(
            self.hyperparams["N_UNLABEL"] * 2).batch(
            self.hyperparams["N_UNLABEL"] * self.hyperparams["BATCH_SIZE"] // (self.hyperparams["N_LABEL"] + self.hyperparams["N_UNLABEL"]))
        self.dataset = tf.data.Dataset.zip((labeled_dataset, unlabeled_dataset))
        
        # load clusters
        self.cluster_centers = tf.convert_to_tensor(get_cluster_centers(self.hyperparams["N_CLASSES"], self.hyperparams["CLUSTERS_DELTA"], self.hyperparams["N_CLUSTERS"], seed=0), dtype=tf.float32)
    
    def _initialize_model(self):
        # get input shapes
        input_shape = self.x_train_labeled.shape[1:]
        output_shape = self.hyperparams["N_CLASSES"]
        
        ################################### INITIALIZE MODEL ###################################
        self.model = SimpleConvNet(input_shape, output_shape)
        self.model_optimizer = k.optimizers.Adam()
        self.model_loss = k.losses.categorical_crossentropy
        
    def _get_groups(self, y_pred):
        y_pred_tiled = tf.tile(tf.reshape(y_pred, (y_pred.shape[0], 1, -1)), (1, self.hyperparams["N_CLUSTERS"], 1))
        cluster_centers_tiled = tf.tile(tf.reshape(self.cluster_centers, (1, self.hyperparams["N_CLUSTERS"], -1)), (y_pred.shape[0], 1, 1))
        
        return tf.cast(tf.argmin(tf.sqrt(tf.reduce_sum((y_pred_tiled - cluster_centers_tiled) ** 2, axis=-1)), axis=1), dtype=tf.int32)
    
    def _get_state(self):
        groups_valid = self._get_groups(self.last_y_val_pred)
        groups_count = tf.pad(tf.math.bincount(groups_valid), [[0, self.hyperparams["N_CLUSTERS"] - tf.reduce_max(groups_valid)-1]])
        groups_correct = tf.pad(tf.math.bincount(groups_valid, weights=tf.cast(tf.argmax(self.y_valid, axis=1) == tf.argmax(self.last_y_val_pred, axis=1), dtype=tf.int32)), [[0, self.hyperparams["N_CLUSTERS"] - tf.reduce_max(groups_valid)-1]])
        
        # np.savetxt("counts_" + str(time.time()), groups_count.numpy())
        # np.savetxt("correct_" + str(time.time()), groups_correct.numpy())
        
        state = tf.where(groups_count > 0, groups_correct / (groups_count + 1), 0)
        return tf.cast(state, dtype=tf.float32).numpy()
    
    def step(self, action):        
        losses = []

        self.last_action = tf.convert_to_tensor(action)
        label_weights = tf.gather(self.last_action, self._get_groups(self.last_y_label_pred))
        unlabel_weights = tf.gather(self.last_action, self._get_groups(self.last_y_unlabel_pred))
        
        # y_max_values = tf.tile(tf.reshape(tf.reduce_max(self.last_y_unlabel_pred, axis=1), (-1, 1)), (1, self.last_y_unlabel_pred.shape[1]))
        # y_train_estimated = tf.where(self.last_y_unlabel_pred == y_max_values, 1.0, 0.0)
        
        for batch, ((x_batch_labeled, y_batch_labeled, labeled_indices), (x_batch_unlabeled, y_batch_unlabeled, unlabeled_indices)) in enumerate(self.dataset):
            
            x_batch = tf.concat([x_batch_labeled, x_batch_unlabeled], axis=0)
            
            y_batch_estimated_label = tf.gather(self.last_y_label_pred, labeled_indices)
            y_batch_estimated_unlabel = tf.gather(self.last_y_unlabel_pred, unlabeled_indices) # = tf.concat([y_batch_labeled, y_batch_unlabeled], axis=1)
            y_batch_estimated = tf.concat([y_batch_estimated_label, y_batch_estimated_unlabel], axis=0)
            
            y_max_values = tf.tile(tf.reshape(tf.reduce_max(y_batch_estimated, axis=1), (-1, 1)), (1, y_batch_estimated.shape[1]))
            y_train_estimated = tf.where(y_batch_estimated == y_max_values, 1.0, 0.0)
            
            with tf.GradientTape() as g:
                """
                    y_batch_labeled_pred = self.model(x_batch_labeled, training=True)
                    loss_labeled = tf.reduce_mean(self.model_loss(y_batch_labeled, y_batch_labeled_pred))
                    
                    y_batch_unlabeled_pred = self.model(x_batch_unlabeled, training=True)
                    batch_weights = tf.gather(weights, unlabeled_indices)
                    y_batch_unlabeled_est = tf.gather(y_train_estimated, unlabeled_indices)
                    loss_unlabeled = tf.reduce_mean(self.model_loss(y_batch_unlabeled_est, y_batch_unlabeled_pred) * batch_weights)
                """
                
                y_batch_pred = self.model(x_batch_unlabeled, training=True)
                loss = tf.reduce_mean(self.model_loss(y_batch_estimated_unlabel, y_batch_pred))
                
                # loss = loss_labeled + loss_unlabeled
                
            gradients = g.gradient(loss, self.model.trainable_variables)
            self.model_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            losses.append(loss.numpy())
            
            if batch >= (self.hyperparams["N_LABEL"] + self.hyperparams["N_UNLABEL"]) // self.hyperparams["BATCH_SIZE"]:
                break
                
        self.last_loss = np.mean(losses)
        
        self.last_y_val_pred = self.model.predict(self.x_valid, batch_size=self.hyperparams["BATCH_SIZE"])
        self.last_y_label_pred == self.model.predict(self.x_train_labeled, batch_size=self.hyperparams["BATCH_SIZE"])
        self.last_y_unlabel_pred = self.model.predict(self.x_train_unlabeled, batch_size=self.hyperparams["BATCH_SIZE"])
        
        acc_t1 = accuracy_score(tf.argmax(self.y_valid, axis=1), tf.argmax(self.last_y_val_pred, axis=1))
        reward = acc_t1 - self.last_accuracy
    
        self.last_accuracy = acc_t1
        self.last_state = observation = self._get_state()
        
        terminal = False if self.timestep < self.hyperparams["N_TIMESTEPS"] or self.last_accuracy < 1.0 else True
        info = {"val_acc": self.last_accuracy, "loss": self.last_loss}
        self.timestep += 1
        
        return observation, reward, terminal, info
    
    def reset(self):
        self._initialize_model()
        
        self.last_loss = float("inf")
        self.last_action = tf.zeros(2)
        self.last_state = tf.zeros(self.hyperparams["N_CLUSTERS"])
        
        if self.hyperparams["N_EPISODES_WARMUP"] > 0:
            for episode in range(self.hyperparams["N_EPISODES_WARMUP"]):
                self.timestep = episode
                
                # self.model.fit(self.x_train_labeled, self.y_train_labeled, epochs=1, verbose=0)
                for batch, ((x_batch_label, y_batch_label, indices_label), _) in enumerate(self.dataset):
                    with tf.GradientTape() as g:
                        y_batch_pred = self.model(x_batch_label, training=True)    
                        loss = tf.reduce_mean(self.model_loss(y_batch_label, y_batch_pred))
                        
                    gradients = g.gradient(loss, self.model.trainable_variables)
                    self.model_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    if batch >= (self.hyperparams["N_LABEL"] + self.hyperparams["N_UNLABEL"]) // self.hyperparams["BATCH_SIZE"]:
                        break
                
                self.last_y_val_pred = self.model.predict(self.x_valid, batch_size=self.hyperparams["BATCH_SIZE"])
                self.last_accuracy = accuracy_score(tf.argmax(self.y_valid, axis=1), tf.argmax(self.last_y_val_pred, axis=1))
                self.render()
        
        self.timestep = 0
        
        self.last_y_val_pred = self.model.predict(self.x_valid, batch_size=self.hyperparams["BATCH_SIZE"])
        self.last_y_unlabel_pred = self.model.predict(self.x_train_unlabeled, batch_size=self.hyperparams["BATCH_SIZE"])
        self.last_y_label_pred = self.model.predict(self.x_train_labeled, batch_size=self.hyperparams["BATCH_SIZE"])
        
        self.last_accuracy = accuracy_score(tf.argmax(self.y_valid, axis=1), tf.argmax(self.last_y_val_pred, axis=1))
        
        self.last_state = self._get_state()
        
        return self.last_state
    
    def render(self, mode="ansi"):
        render_string = ""
        
        render_string += "TIMESTEP: %d - LOSS: %.3f - VAL_ACC: %.3f" % (self.timestep, self.last_loss, self.last_accuracy)
        
        render_string += "\n\naction: %.3f (+/- %.3f)" % (np.mean(self.last_action), np.std(self.last_action))
        render_string += "\n" + str(self.last_action.numpy())
        
        render_string += "\n\nstate (min, max): %.3f (%d), %.3f (%d)" % (
            np.min(self.last_state), np.argmin(self.last_state), np.max(self.last_state), np.argmax(self.last_state)
        )
        render_string += "\n" + str(self.last_state)
        
        # print("Mode should be ANSI. Printing in render():", render_string, end="\r")
        print(render_string, file=open("/opt/workspace/host_storage_hdd/tmp.log", "w"))
        
        return render_string
    