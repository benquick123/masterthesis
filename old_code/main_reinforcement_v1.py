import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
import datetime
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment    

from sklearn.cluster import MiniBatchKMeans, KMeans

from model_WN import MeanTeacherModelWN
from model_BN import MeanTeacherModelBN
from model import SimpleConvNet
from utils import data_generator, Log, get_image_augmentations, rampdown, rampup
from preprocess import preprocess, SVHN
import reinforcement as ddpg

import time

"""
def minimize_distance(s0, s1):
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    distances = np.zeros((s0.shape[0], s1.shape[0]))
    for i in range(s0.shape[0]):
        for j in range(s1.shape[1]):
            distances[i, j] = euclidean_distance(s0[i], s1[j])
    
    s0_sorted_indices, s1_sorted_indices = linear_sum_assignment(distances)
    assert all(s0_sorted_indices[:-1] < s0_sorted_indices[1:])
    
    s1_sorted = s1[s1_sorted_indices]
    
    return s1_sorted, s1_sorted_indices
"""

"""
def _get_state(model, prev_state=None):
    # kmeans = KMeans(n_clusters=N_CLUSTERS, n_jobs=1, init=prev_state if prev_state is not None else "k-means++")
    p0 = model.predict(x_train_unlabeled, batch_size=BASE_BATCH)
    groups = kmeans.fit_predict(p0)
    new_state = kmeans.cluster_centers_
    
    if prev_state is not None:
        new_state, indices = minimize_distance(prev_state, new_state)
        # assert all(indices == np.arange(len(indices)))
        inv_indices = np.arange(len(indices))[np.argsort(indices)]
        groups = np.array([inv_indices[g] for g in groups])
    
    return new_state, p0, groups
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    state = tf.zeros(N_CLUSTERS)
    distances = np.zeros((N_VALID, N_CLUSTERS))
    y_pred_valid = tf.Variable(model.predict(x_valid, batch_size=BASE_BATCH))

    for i, y in enumerate(y_pred_valid):
        for j, cluster_center in enumerate(cluster_centers):
            distances[i, j] = euclidean_distance(y, cluster_center)
    
    valid_groups = tf.argmin(distances, axis=1)
    for i in range(N_CLUSTERS):
        in_cluster = i == valid_groups
        if np.any(in_cluster):
            state[i] = accuracy_score(tf.argmax(y_valid[in_cluster], axis=1), tf.argmax(y_pred_valid[in_cluster], axis=1))
    
    p0 = model.predict(x_train_unlabeled, batch_size=BASE_BATCH)
    distances = np.zeros((N_UNLABEL, N_CLUSTERS))
    for i, y in enumerate(p0):
        for j, cluster_center in enumerate(cluster_centers):
            distances[i, j] = euclidean_distance(y, cluster_center)
    
    groups = tf.argmin(distances, axis=1)
"""


def get_state(y_pred_unlabel, y_pred_valid):
    y_pred_valid_tiled = tf.tile(tf.reshape(y_pred_valid, (N_VALID, 1, -1)), (1, N_CLUSTERS, 1))
    cluster_centers_tiled = tf.tile(tf.reshape(cluster_centers, (1, N_CLUSTERS, -1)), (N_VALID, 1, 1))
    
    groups_valid = tf.cast(tf.argmin(tf.sqrt(tf.reduce_sum((y_pred_valid_tiled - cluster_centers_tiled) ** 2, axis=-1)), axis=1), dtype=tf.int32)
    groups_count = tf.pad(tf.math.bincount(groups_valid), [[0, N_CLUSTERS-tf.reduce_max(groups_valid)-1]])
    groups_correct = tf.pad(tf.math.bincount(groups_valid, weights=tf.cast(tf.argmax(y_valid, axis=1) == tf.argmax(y_pred_valid, axis=1), dtype=tf.int32)), [[0, N_CLUSTERS-tf.reduce_max(groups_valid)-1]])
    
    state = tf.where(groups_count > 0, groups_correct / groups_count, 0)
    
    y_pred_unlabel_tiled = tf.tile(tf.reshape(y_pred_unlabel, (N_UNLABEL, 1, -1)), (1, N_CLUSTERS, 1))
    cluster_centers_tiled = tf.tile(tf.reshape(cluster_centers, (1, N_CLUSTERS, -1)), (N_UNLABEL, 1, 1))
    
    groups = tf.cast(tf.argmin(tf.sqrt(tf.reduce_sum((y_pred_unlabel_tiled - cluster_centers_tiled) ** 2, axis=-1)), axis=1), dtype=tf.int32)
    
    return tf.cast(state, dtype=tf.float32), groups
    

def train():
    replay_buffer = ddpg.ReplayBuffer(BUFFER_SIZE)
    
    for episode in range(N_EPISODES):
        Log._print("EPISODE: %d" % episode)
        
        base_model.reset()
        # base_model = SimpleConvNet(input_shape, output_shape)
        # base_model.compile(base_model_optimizer, loss=base_model_loss)
        
        """for epoch in range(1):
            # x_train_labeled_aug = get_image_augmentations(x_train_labeled, BATCH_SIZE=len(x_train_labeled))[0]
            loss = base_model.fit(x_train_labeled, y_train_labeled, batch_size=BASE_BATCH, epochs=1, verbose=0)
            Log._print("pretraining - epoch: %d - loss: %.3f" % (epoch, loss.history["loss"][0]), end="\r")
        print()"""
        
        # p0 = base_model.predict(x_train_unlabeled, batch_size=BASE_BATCH)
        """y_pred_valid_0 = base_model.predict(x_valid, batch_size=BASE_BATCH)
        s0, g0 = get_state(p0, y_pred_valid_0)"""
        
        ACC_t0 = 0      # accuracy_score(tf.argmax(y_valid, axis=1), tf.argmax(y_pred_valid_0, axis=1))
        Log._print("base accuracy after init train: %.3f" % (ACC_t0))

        # logging
        accumulated_reward = 0
        
        for timestep in range(N_TIMESTEPS):
            #   p0 = base_model.predict(x_train_unlabeled, batch_size=BASE_BATCH)
            
            # a = tf.reshape(actor.get_action_with_noise(tf.reshape(s0, (1, -1))), (-1, ))
            # weights = tf.gather(a, g0)
            # weights = tf.zeros(N_UNLABEL)
            
            # y_max_values = tf.tile(tf.reshape(tf.reduce_max(p0, axis=1), (-1, 1)), (1, p0.shape[1]))
            # y_train_estimated = tf.where(p0 == y_max_values, 1.0, 0.0)

            for batch, ((x_batch_labeled, y_batch_labeled, labeled_indices), (x_batch_unlabeled, y_batch_unlabeled, unlabeled_indices)) in enumerate(dataset):
                """X = tf.concat([x_batch_labeled, x_batch_unlabeled], axis=0)
                y = tf.concat([y_batch_labeled, tf.gather(y_train_estimated, unlabeled_indices)], axis=0)
                batch_weights = tf.concat([tf.ones(len(labeled_indices)), tf.gather(weights, unlabeled_indices)], axis=0)"""

                # base_model.train_on_batch(X, y, sample_weight=batch_weights)
                base_model.train_on_batch(x_batch_labeled, y_batch_labeled)
                
                if batch >= (N_LABEL + N_UNLABEL) // BASE_BATCH:
                    break
            
            # p1 = base_model.predict(x_train_unlabeled, batch_size=BASE_BATCH)
            """y_pred_valid_1 = base_model.predict(x_valid, batch_size=BASE_BATCH)
            s1, g1 = get_state(p1, y_pred_valid_1)"""

            """ACC_t1 = accuracy_score(tf.argmax(y_valid, axis=1), tf.argmax(y_pred_valid_1, axis=1))
            r = ACC_t1 - ACC_t0
            t = (timestep + 1) == N_TIMESTEPS"""

            """replay_buffer.add(tf.reshape(s0, (-1, )), a, r, t, tf.reshape(s1, (-1, )))
            if len(replay_buffer) > RL_BATCH:
                samples = replay_buffer.sample_batch(RL_BATCH)
                ddpg.ddpg_update(samples, actor, critic, GAMMA)"""

            # accumulated_reward += r
            accumulated_reward = 0
            r = 0
            a = np.zeros(10)
            ACC_t1 = 0
            Log._print("timestep: %02d - base_acc: %.3f - accum_reward: %.5f - curr_reward: %.5f - action: %.3f (+/- %.3f)        " % (
                timestep, ACC_t1, accumulated_reward, r, np.mean(a), np.std(a)
            ), end="\r")
            
            # s0, p0, g0, ACC_t0, y_pred_valid_0 = s1, p1, g1, ACC_t1, y_pred_valid_1
            # p0 = p1

        print()

if __name__ == "__main__":
    # logging initialization
    answer = input("Log?[y/N] ")
    if answer in set(["y", "yes", "Y", "YES", "YA", "YISS"]):
        Log.log = True
        Log.current_time = str(datetime.datetime.now()).replace(":", "-").split(".")[0]
        Log.save_self()
    
    # common hyperparams
    N_EPISODES = 500
    N_TIMESTEPS = 50
    N_LABEL = 750
    N_UNLABEL = 5000
    N_VALID = 250
    BASE_BATCH = 100
    RL_BATCH = 128
    N_CLUSTERS = 50
    
    # RL hyperparams
    CRITIC_LR = 0.001
    ACTOR_LR = 0.0001
    TAU = 0.001
    GAMMA = 0.99
    BUFFER_SIZE = 2**16
    
    ################################## DATA LOADING ####################################
    # load data
    data_filepath = "/opt/workspace/host_storage_hdd/"
    (x_train, y_train), (x_test, y_test) = SVHN.load_data(path=data_filepath)
    (x_train_labeled, y_train_labeled), (x_train_unlabeled, y_train_unlabeled), (x_valid, y_valid) = preprocess(x_train, y_train, N_LABEL, N_UNLABEL, N_VALID)
    
    # initialize tensorflow dataset
    labeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[x_train_labeled, y_train_labeled]).shuffle(N_LABEL*2).batch(N_LABEL * BASE_BATCH // (N_LABEL + N_UNLABEL))
    unlabeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[x_train_unlabeled, y_train_unlabeled]).shuffle(N_UNLABEL*2).batch(N_UNLABEL * BASE_BATCH // (N_LABEL + N_UNLABEL))
    dataset = tf.data.Dataset.zip((labeled_dataset, unlabeled_dataset))
    
    # get input shapes
    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1]
    
    state_space = N_CLUSTERS
    action_space = N_CLUSTERS
    
    ############################## INITIALIZE MODELS ###################################
    base_model = SimpleConvNet(input_shape, output_shape)
    base_model_optimizer = k.optimizers.Adam()
    base_model_loss = k.losses.CategoricalCrossentropy()
    base_model.compile(base_model_optimizer, loss=base_model_loss)
    
    critic = ddpg.CriticNetwork(state_space, action_space, TAU, optimizer=k.optimizers.Adam(CRITIC_LR))
    actor = ddpg.ActorNetwork(state_space, action_space, TAU, optimizer=k.optimizers.Adam(ACTOR_LR), noise_sigma=0.075, action_range=(0, 1))
    
    # kmeans = KMeans(n_clusters=N_CLUSTERS, n_jobs=-1)
    cluster_centers = tf.convert_to_tensor(np.load("/opt/workspace/host_storage_hdd/cluster_centers_" + str(N_CLUSTERS) + ".npy"), dtype=tf.float32)
    
    train()