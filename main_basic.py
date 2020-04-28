import warnings
warnings.filterwarnings('ignore')

gpu_num = '0,1,2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
import errno

import time
import argparse
from datetime import datetime
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
import shutil

import torch
import torch.multiprocessing as mp
from torchvision import transforms

from multiprocessing import Process
from multiprocessing.managers import BaseManager

from env import SelfTeachingBaseEnv
from sac_multi import SAC_Trainer, ReplayBuffer, share_parameters, worker
from utils import learn_callback, test_pipeline, Logger


if __name__ == '__main__':
    os.system('clear')
    
    parser = argparse.ArgumentParser(description="Runner for one dataset pipeline.")
    
    # general
    general_group = parser.add_argument_group("general")
    general_group.add_argument("--dataset", type=str, action="store", required=True, help="Dataset to use.")
    general_group.add_argument("--num-workers", type=int, default=1, action="store", help="Number of workers to initialize and run the experiments with.")
    general_group.add_argument("--path-postfix", type=str, default="", action="store", help="Path postfix that is added to the logging folder.")
    general_group.add_argument("--config-path", type=str, default="./config", action="store", help="Folder containing config file.")
    
    # testing
    testing_group = parser.add_argument_group("testing")
    testing_group.add_argument("--test", action="store_true", help="Run in test mode?")
    testing_group.add_argument("--new-folder", action="store_true", help="Whether to create new folder when running. Only has an effect when testing.")
    
    # training
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--agent-rl", type=float, default=5e-5, action="store", help="SAC agent learning rate.")
    training_group.add_argument("--batch-size", type=int, default=16, action="store", help="Batch size for SAC trainer.")
    training_group.add_argument("--buffer-size", type=int, default=200000, action="store", help="Size of the replay buffer.")
    training_group.add_argument("--rl-hidden-layer-sizes", type=int, nargs="+", default=[128, 128], help="Sizes of hidden layers.")
    training_group.add_argument("--num-steps", type=int, default=400000, action="store", help="Number of steps for RL training. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--n-warmup", type=int, default=20000, action="store", help="Number of timesteps before agent's actions are used. Random actions are used before that. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--learning-starts", type=int, default=20000, action="store", help="Number of timesteps before RL agents starts training. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--start-step", type=int, default=0, action="store", help="At which step to start training. Useful when learning from pretrained.")
    training_group.add_argument("--test-interval", type=int, default=70, action="store", help="RL agent test interval. Divided by args.num_workers when multiprocessing is enabled.")
    
    # transfer
    transfer_group = parser.add_argument_group("transfer")
    transfer_group.add_argument("--from-dataset", type=str, default="", action="store", help="Dataset name to transfer from.")
    transfer_group.add_argument("--pretrained-path", type=str, default="", action="store", help="In test mode, model from this path is loaded.")
    transfer_group.add_argument("--load-buffer", action="store_true", help="Whether to load 'replay_buffer' from pretrained-path. Only has effect when --test is False and --pretrained-path is specified.")
    transfer_group.add_argument("--load-model", action="store_true", help="Whether to load agent's weights from pretrained-path. Only has effect when --test is False and --pretrained-path is specified.")
    
    args = parser.parse_args()

    """
        N_CLUSTERS = 1
        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                            ElasticTransform(5, 35),
                                            transforms.RandomChoice([transforms.RandomAffine(7.5), transforms.RandomAffine(0, shear=7.5)]),
                                            transforms.RandomSizedCrop(28, scale=(0.8, 1.2)),
                                            transforms.ToTensor()])
    """
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(args.buffer_size)
    
    logger = Logger()
    
    env_kwargs = {"logger": logger,
                  "config_path": args.config_path,
                  "dataset": args.dataset, 
                  "override_hyperparams": dict()
                  }
    
    if args.from_dataset != "":
        env = SelfTeachingBaseEnv(logger=logger, config_path=args.config_path, dataset=args.from_dataset)
        env_kwargs['override_hyperparams']['output_state_dim'] = env.hyperparams['output_state_dim']
        env.close()
        del env

    env = SelfTeachingBaseEnv(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, logger, hidden_layer_sizes=args.rl_hidden_layer_sizes, q_lr=args.agent_rl, pi_lr=args.agent_rl, alpha_lr=args.agent_rl, v_lr=args.agent_rl)
    
    if args.test:
        if args.pretrained_path == "":
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "Pretrained model path not specified.")
        
        if args.new_folder:
            save_path = args.pretrained_path + ("_to_" + args.dataset if args.dataset != "" else "") + ("_" + args.path_postfix if args.path_postfix != "" else "") + "_test"
            logger.set_path(save_path=save_path)
            logger.create_logdirs(args, save_self=False)
        else:
            # this will overwrite old results.
            logger.set_path(save_path=args.pretrained_path)
            logger.print("Overwriting old results.")
            
        env_kwargs['override_hyperparams']['reward_history_threshold'] = -10.0
        test_pipeline(env, sac_trainer, logger, model_path=args.pretrained_path)

    else:
        env.close()
        del env
        
        if args.pretrained_path != "":
            if args.load_buffer:
                replay_buffer.load_buffer(args.pretrained_path)
            if args.load_model:
                sac_trainer.load_model(os.path.join(args.pretrained_path, 'best_by_test_sac_self_teaching'))
        
        logger.create_logdirs(args)

        sac_trainer.q_net_1.share_memory()
        sac_trainer.q_net_2.share_memory()
        sac_trainer.policy_net.share_memory()
        sac_trainer.value_net.share_memory()
        sac_trainer.target_value_net.share_memory()
        sac_trainer.log_alpha.share_memory()
        share_parameters(sac_trainer.q_optimizer_1)
        share_parameters(sac_trainer.q_optimizer_2)
        share_parameters(sac_trainer.policy_optimizer)
        share_parameters(sac_trainer.alpha_optimizer)
        share_parameters(sac_trainer.v_optimizer)
        
        processes = []
        for i in range(args.num_workers):
            process = Process(target=worker,
                            kwargs={'worker_id': i,
                                    'sac_trainer': sac_trainer,
                                    'env_fn': SelfTeachingBaseEnv,
                                    'env_kwargs': env_kwargs,
                                    'replay_buffer': replay_buffer,
                                    'num_steps': args.num_steps // args.num_workers,
                                    'learning_starts': args.learning_starts // args.num_workers,
                                    'n_warmup': args.n_warmup // args.num_workers,
                                    'start_step': args.start_step,
                                    # 'linear_lr_scheduler': [args.agent_rl, final_lr],
                                    'n_updates': args.num_workers,
                                    'batch_size': args.batch_size,
                                    'callback': learn_callback,
                                    'callback_kwargs': {'test_interval': args.test_interval // args.num_workers},
                                    'logger': logger}
                            )
            process.daemon = True
            processes.append(process)
            
        for process in processes:
            process.start()
            # sleep for a few seconds to ensure enough memory is left.
            time.sleep(5)
            
        for process in processes:
            process.join()
        
        replay_buffer.save_buffer(logger.save_path)
        
        env_kwargs['override_hyperparams']['reward_history_threshold'] = -10.0
        test_pipeline(SelfTeachingBaseEnv(**env_kwargs), sac_trainer, logger)

