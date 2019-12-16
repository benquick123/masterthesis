from tensorflow import keras as k
from collections import deque

import tensorflow as tf
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size=1024):
        self.buffer_size = buffer_size
        self.buffer = deque()
    
    def add(self, s0, a, r, t, s1):
        to_add = (s0, a, r, t, s1)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(to_add)
        else:
            self.buffer.popleft()
            self.buffer.append(to_add)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indices = np.random.permutation(np.arange(len(self.buffer)))[:batch_size]
        
        s0, a, r, t, s1 = [], [], [], [], []
        for i in indices:
            s0.append(self.buffer[i][0])
            a.append(self.buffer[i][1])
            r.append(self.buffer[i][2])
            t.append(self.buffer[i][3])
            s1.append(self.buffer[i][4])
        
        r = tf.reshape(r, (-1, 1))
        return tf.convert_to_tensor(s0), tf.convert_to_tensor(a), tf.convert_to_tensor(r), tf.convert_to_tensor(t), tf.convert_to_tensor(s1)

    def clear(self):
        self.buffer = deque()


class ActorNetwork(object):

    def __init__(self, state_space, action_space, tau, noise_mean=0, noise_sigma=0.2, optimizer=k.optimizers.Adam(), action_range=(0, 1), layer_size=(400, 300)):        
        input0, output0 = self.create_model(state_space, action_space, layer_size)
        self.model = k.models.Model(input0, output0)

        input0, output0 = self.create_model(state_space, action_space, layer_size)
        self.target_model = k.models.Model(input0, output0)
        self.target_model.set_weights(self.model.get_weights())

        self.tau = tau
        self.optimizer = optimizer
        self.action_range = action_range
        self.default_range = (-1, 1)

        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma

    
    def create_model(self, state_space, action_space, layer_size):
        input0 = k.layers.Input(state_space)

        x = k.layers.Dense(layer_size[0])(input0)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)

        x = k.layers.Dense(layer_size[1])(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)

        kernel_initializer = k.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output0 = k.layers.Dense(action_space, activation="tanh", kernel_initializer=kernel_initializer)(x)

        return input0, output0

    def fit(self, s, critic):
        with tf.GradientTape() as g:
            a = self.predict(s)
            v = critic.predict([s, a])
            loss = -tf.reduce_mean(v)
            
        gradients = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def predict(self, s, training=False):
        return self.scale_action(self.model(s, training=training))

    def predict_target(self, s, training=False):
        return self.scale_action(self.target_model(s, training=training))

    def update_target(self):
        weights = self.model.weights
        target_weights = self.target_model.weights

        for layer_w, target_layer_w in zip(weights, target_weights):
            target_layer_w.assign(self.tau * layer_w + (1 - self.tau) * target_layer_w)

    def scale_action(self, a):
        original_scope = self.default_range[1] - self.default_range[0]
        target_scope = self.action_range[1] - self.action_range[0]
        return (((a - self.default_range[0]) * target_scope) / original_scope) + self.action_range[0]

    @tf.function
    def get_action_with_noise(self, s, mean=None, sigma=None):
        if mean is None:
            mean = self.noise_mean
        if sigma is None:
            sigma = self.noise_sigma
            
        action = self.predict(s)
        action += tf.random.normal(shape=action.shape, mean=mean, stddev=sigma, dtype=tf.float32)

        return tf.clip_by_value(action, *self.action_range)


class CriticNetwork(object):

    def __init__(self, state_space, action_space, tau, optimizer=k.optimizers.Adam(), layer_size=(400, 300)):
        inputs, output0 = self.create_model(state_space, action_space, layer_size)
        self.model = k.models.Model(inputs, output0)

        inputs, output0 = self.create_model(state_space, action_space, layer_size)
        self.target_model = k.models.Model(inputs, output0)
        self.target_model.set_weights(self.model.get_weights())

        self.tau = tau
        self.optimizer = optimizer

    def create_model(self, state_space, action_space, layer_size):
        input0 = k.layers.Input(state_space)
        input1 = k.layers.Input(action_space)
        
        x = k.layers.Concatenate(axis=1)([input0, input1])

        x = k.layers.Dense(layer_size[0])(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        
        x = k.layers.Dense(layer_size[1])(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)

        kernel_initializer = k.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output0 = k.layers.Dense(1, kernel_initializer=kernel_initializer)(x)

        return [input0, input1], output0

    def fit(self, state_input, y):
        model_loss = k.losses.MeanSquaredError()
        with tf.GradientTape() as g:
            y_pred = self.predict(state_input)
            loss = model_loss(y, y_pred)

        gradients = g.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def predict(self, state_input, training=False):
        return self.model(state_input, training=training)

    def predict_target(self, state_input, training=False):
        return self.target_model(state_input, training=training)

    def update_target(self):
        weights = self.model.weights
        target_weights = self.target_model.weights

        for layer_w, target_layer_w in zip(weights, target_weights):
            # weight update is implicit
            target_layer_w.assign(self.tau * layer_w + (1 - self.tau) * target_layer_w)
        

@tf.function
def ddpg_update(samples, actor, critic, GAMMA):
    s0, a, r, t, s1 = samples
    if len(a.shape) != 2:
        a = tf.reshape(a, (-1, 1))
        
    # get q and target ys
    target_q = critic.predict_target([s1, actor.predict_target(s1)])
    t = tf.reshape(1.0 - tf.cast(t, tf.float32), (-1, 1))
    r = tf.cast(r, tf.float32)
    yi = r + t * GAMMA * target_q

    # fit critic & actor
    critic_loss = critic.fit([s0, a], yi)
    actor_loss = actor.fit(s0, critic)

    # update target networks
    critic.update_target()
    actor.update_target()
    
@tf.function
def get_rewards(y_pred_unlabel, y_pred_valid, losses_delta):
    # reshape tensors
    y_pred_unlabel = tf.reshape(y_pred_unlabel, (y_pred_unlabel.shape[0], 1, -1))
    y_pred_valid = tf.reshape(y_pred_valid, (1, y_pred_valid.shape[0], -1))
    
    # tile tensors to the same size
    y_pred_unlabel = tf.tile(y_pred_unlabel, (1, y_pred_valid.shape[1], 1))
    y_pred_valid = tf.tile(y_pred_valid, (y_pred_unlabel.shape[0], 1, 1))
    
    # compute inverse ED and normalize
    inverse_ED = 1 - tf.sqrt(tf.reduce_sum(tf.pow(y_pred_unlabel - y_pred_valid, 2), axis=-1))
    inverse_ED_normalized = inverse_ED / tf.reshape(tf.reduce_sum(inverse_ED, axis=1), (-1, 1))
    inverse_ED_normalized = tf.clip_by_value(inverse_ED_normalized, 0.0, 1.0)
    
    # prepare losses and get rewards
    losses_delta = tf.tile(tf.reshape(losses_delta, (1, -1)), (y_pred_unlabel.shape[0], 1))
    r = tf.reduce_sum(inverse_ED_normalized * losses_delta, axis=1)
    return r