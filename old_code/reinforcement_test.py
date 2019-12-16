import reinforcement as ddpg

import numpy as np
import tensorflow as tf
import gym

from tensorflow import keras as k


def train():
    # Initialize replay memory
    replay_buffer = ddpg.ReplayBuffer(BUFFER_SIZE)
    
    training_started = False
    
    for i in range(N_EPOCH):
        ep_reward = 0
        s = env.reset()

        for j in range(MAX_EPISODE_LEN):
            env.render()

            # Added exploration noise
            a = actor.get_action_with_noise(np.reshape(s, (1, state_space)))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (state_space,)), np.reshape(a, (action_space,)), float(r),
                              terminal, np.reshape(s2, (state_space,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if len(replay_buffer) > BATCH_SIZE:
                if not training_started:
                    training_started = True
                    print("TRAINING STARTED.")
                samples = replay_buffer.sample_batch(BATCH_SIZE)
                ddpg.ddpg_update(samples, actor, critic, GAMMA)

            s = s2
            ep_reward += r
            print("episode: %d - step: %d - accumulated reward: %f - replay buffer: %d - training: %d" % (i, j, ep_reward, len(replay_buffer), training_started), end="\r")

            if terminal:
                print()
                break


if __name__ == "__main__":
    CRITIC_LR = 0.001
    ACTOR_LR = 0.001
    TAU = 0.005
    GAMMA = 0.99
    BUFFER_SIZE = 2**20
    
    BATCH_SIZE = 100
    N_EPOCH = 30
    MAX_EPISODE_LEN = 200
    
    env = gym.make("Pendulum-v0")
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    
    # initialize critic & actor
    critic = ddpg.CriticNetwork(state_space, action_space, TAU, optimizer=k.optimizers.Adam(CRITIC_LR))
    actor = ddpg.ActorNetwork(state_space, action_space, TAU, optimizer=k.optimizers.Adam(ACTOR_LR), noise_sigma=0.1, action_range=(-2, 2))
    
    train()
    
    