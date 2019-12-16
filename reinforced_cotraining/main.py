import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

from cotraining_env import CotrainEnv

if __name__ == "__main__":
    parser = Trainer.get_argument()
    parser = DQN.get_argument(parser)
    
    args = parser.parse_args()
    
    env = CotrainEnv(N_CLUSTERS=120, N_EPISODES_WARMUP=1)
    test_env = CotrainEnv(N_CLUSTERS=120, N_EPISODES_WARMUP=1)
    policy = DQN(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        enable_categorical_dqn=args.enable_categorical_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        target_replace_interval=300,
        discount=0.99,
        gpu=0,
        memory_capacity=1e4,
        batch_size=32,
        n_warmup=500)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
    
    