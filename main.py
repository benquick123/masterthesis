gpu_num = '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
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

from env import SelfTeachingEnv

# parser = OnPolicyTrainer.get_argument()
parser = Trainer.get_argument()
# parser = DDPG.get_argument(parser)
parser = SAC.get_argument(parser)
# parser = PPO.get_argument(parser)
args = parser.parse_args()

print("Initializing: env")
env = SelfTeachingEnv(EPOCHS_PER_STEP=2, SIGNIFICANCE_DECAY=0.1, ID=gpu_num)
print("Initializing: test_env")
test_env = SelfTeachingEnv(EPOCHS_PER_STEP=2, SIGNIFICANCE_DECAY=0.1, ID=gpu_num)

policy = SAC(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    memory_capacity=1e5,
    min_action=env.action_space.low,
    max_action=env.action_space.high,
    batch_size=128,
    auto_alpha=args.auto_alpha,
    n_warmup=1024,
    actor_units=[128, 192, 256, 128],
    critic_units=[128, 192, 256, 128],
    lr=3e-4
) 

"""
policy = PPO(
    state_shape=env.observation_space.shape,
    action_dim=get_act_dim(env.action_space),
    is_discrete=False,
    min_action=env.action_space.low,
    max_action=env.action_space.high,
    batch_size=64,
    actor_units=[128, 64],
    critic_units=[128, 64],
    n_epoch=5,
    n_epoch_critic=5,
    lr_actor=3e-4,
    lr_critic=3e-4,
    discount=0.99,
    lam=0.95,
    entropy_coef=0.1,
    horizon=512,
    normalize_adv=args.normalize_adv,
    enable_gae=args.enable_gae,
)"""

trainer = Trainer(policy, env, args, test_env=test_env)
print("Starting training.")
trainer()

"""
command to run:
python3.7 main.py --max-steps 50000 --show-progress --show-test-progress --test-interval 4096
"""