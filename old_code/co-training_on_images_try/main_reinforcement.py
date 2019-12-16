import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

import tensorflow as tf
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.sac import SAC
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim

from reinforcement_env import CurriculumEnv

parser = OnPolicyTrainer.get_argument()
# parser = DDPG.get_argument(parser)
# parser = SAC.get_argument(parser)
parser = PPO.get_argument(parser)
args = parser.parse_args()

env = CurriculumEnv(N_CLASSES=3, N_LABEL=99, N_VALID=990, N_UNLABEL=1500, N_CLUSTERS=20, CLUSTER_DELTA=0.1, BATCH_SIZE=304, N_TIMESTEPS=args.episode_max_steps, n_episodes_warmup=40)
test_env = CurriculumEnv(N_CLASSES=3, N_LABEL=99, N_VALID=990, N_UNLABEL=1500, N_CLUSTERS=20, CLUSTER_DELTA=0.1, BATCH_SIZE=304, N_TIMESTEPS=args.episode_max_steps, n_episodes_warmup=40)

"""
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=1,
    memory_capacity=10000,
    max_action=env.action_space.high[0],
    batch_size=32,
    n_warmup=250,
    lr_actor=0.0001)
"""

"""
policy = SAC(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=0,
    memory_capacity=args.memory_capacity,
    min_action=0.0,
    max_action=1.0,
    batch_size=100,
    n_warmup=250,
    auto_alpha=args.auto_alpha)"""
    
policy = PPO(
    state_shape=env.observation_space.shape,
    action_dim=get_act_dim(env.action_space),
    is_discrete=False,
    min_action=0.0,
    max_action=1.0,
    batch_size=64,
    actor_units=[64, 128],
    critic_units=[64, 128],
    n_epoch=10,
    n_epoch_critic=10,
    lr_actor=3e-4,
    lr_critic=3e-4,
    discount=0.9,
    lam=0.95,
    horizon=512,
    normalize_adv=args.normalize_adv,
    enable_gae=args.enable_gae,
    gpu=0
)
trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
trainer()
