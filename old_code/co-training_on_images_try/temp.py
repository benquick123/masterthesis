import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
import gym

from tf2rl.algos.ppo import PPO
from tf2rl.policies.categorical_actor import CategoricalActorCritic
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="Pendulum-v0")
    parser.set_defaults(test_interval=20480)
    parser.set_defaults(max_steps=int(1e7))
    parser.set_defaults(horizon=2048)
    parser.set_defaults(batch_size=64)
    parser.set_defaults(gpu=0)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[64, 64],
        critic_units=[64, 64],
        n_epoch=10,
        n_epoch_critic=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        discount=0.99,
        lam=0.95,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
"""

import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.dqn import DQN
from tf2rl.algos.ppo import PPO
from tf2rl.algos.vpg import VPG
from tf2rl.envs.utils import is_discrete, get_act_dim
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer

parser = OnPolicyTrainer.get_argument()
# parser = Trainer.get_argument()
# parser = DDPG.get_argument(parser)
parser = PPO.get_argument(parser)
# parser = VPG.get_argument(parser)

parser.set_defaults(test_interval=20480)
parser.set_defaults(max_steps=int(1e7))
parser.set_defaults(horizon=2048)
parser.set_defaults(batch_size=64)

args = parser.parse_args()

# env = gym.make("Pendulum-v0")

# env = gym.make("CartPole-v0")
env = gym.make("MountainCarContinuous-v0")

"""
policy = DQN(
    enable_double_dqn=True,
    enable_dueling_dqn=False,
    enable_noisy_dqn=False,
    enable_categorical_dqn=False,
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.n,
    target_replace_interval=300,
    discount=0.99,
    gpu=0,
    memory_capacity=1000,
    batch_size=256,
    n_warmup=500)
"""

policy = PPO(
    state_shape=env.observation_space.shape,
    action_dim=get_act_dim(env.action_space),
    is_discrete=is_discrete(env.action_space),
    max_action=None if is_discrete(
        env.action_space) else env.action_space.high[0],
    batch_size=args.batch_size,
    actor_units=[64, 64],
    critic_units=[64, 64],
    n_epoch=10,
    n_epoch_critic=10,
    lr_actor=3e-4,
    lr_critic=3e-4,
    discount=0.99,
    lam=0.95,
    horizon=args.horizon,
    normalize_adv=args.normalize_adv,
    enable_gae=args.enable_gae,
    gpu=0
) 

"""
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=0,
    memory_capacity=10000,
    max_action=env.action_space.high[0],
    batch_size=32,
    n_warmup=500,
    min_action=-1.)
"""
# trainer = Trainer(policy, env, args, test_env=test_env)
trainer = OnPolicyTrainer(policy, env, args)
trainer()