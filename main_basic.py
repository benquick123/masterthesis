import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

GPU_NUM = '3,2,1' # np.random.choice(['1', '3'])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
import errno

import time
import argparse

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
    general_group.add_argument("--results-folder", type=str, default="/opt/workspace/host_storage_hdd/results", action="store", help="Folder used as base when creating a results directory.")
    general_group.add_argument("--random-seed", type=int, default=np.random.randint(100000), action="store", help="Random seed used for experiment initalization.")
    
    # testing
    testing_group = parser.add_argument_group("testing")
    testing_group.add_argument("--test", action="store_true", help="Run in test mode?")
    testing_group.add_argument("--new-folder", action="store_true", help="Whether to create new folder when running. Only has an effect when testing.")
    testing_group.add_argument("--manual-thresholds", type=float, nargs="+", default=False, action="store", help="Optional manual thresholds to use during testing.")
    
    # training
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--agent-lr", type=float, default=5e-5, action="store", help="SAC agent learning rate.")
    training_group.add_argument("--batch-size", type=int, default=16, action="store", help="Batch size for SAC trainer.")
    training_group.add_argument("--buffer-size", type=int, default=200000, action="store", help="Size of the replay buffer.")
    training_group.add_argument("--rl-hidden-layer-sizes", type=int, nargs="+", default=[128, 128], help="Sizes of hidden layers.")
    training_group.add_argument("--num-steps", type=int, default=400000, action="store", help="Number of steps for RL training. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--n-warmup", type=int, default=20000, action="store", help="Number of timesteps before agent's actions are used. Random actions are used before that. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--learning-starts", type=int, default=20000, action="store", help="Number of timesteps before RL agents starts training. Divided by args.num_workers when multiprocessing is enabled.")
    training_group.add_argument("--start-step", type=int, default=0, action="store", help="At which step to start training. Useful when learning from pretrained.")
    training_group.add_argument("--test-interval", type=int, default=50, action="store", help="RL agent test interval. Number specifies Divided by args.num_workers when multiprocessing is enabled.")
    
    # transfer
    transfer_group = parser.add_argument_group("transfer")
    transfer_group.add_argument("--from-dataset", type=str, default="", action="store", help="Dataset name to transfer from.")
    transfer_group.add_argument("--pretrained-path", type=str, default="", action="store", help="In test mode, model from this path is loaded. In train mode, this path is used to train the model from.")
    transfer_group.add_argument("--num-buffer-delete", type=int, default=0, action="store", help="Specifies how many experience points should be deleted, when loading from pretrained-path.")
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
                  "override_hyperparams": {
                      "random_seed": args.random_seed
                  }
                  }
    
    if args.from_dataset != "":
        env = SelfTeachingBaseEnv(logger=logger, config_path=args.config_path, dataset=args.from_dataset)
        env_kwargs['override_hyperparams']['output_state_dim'] = env.hyperparams['output_state_dim']
        env.close()
        del env

    env = SelfTeachingBaseEnv(**env_kwargs)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, logger, hidden_layer_sizes=args.rl_hidden_layer_sizes, q_lr=args.agent_lr, pi_lr=args.agent_lr, alpha_lr=args.agent_lr, v_lr=args.agent_lr)
    
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
        test_pipeline(env, sac_trainer, logger, model_path=args.pretrained_path, all_samples=False, manual_thresholds=args.manual_thresholds, labeled_samples=False, all_samples_labeled=False, trained_model=True, conf_matrix=False)
        # test_pipeline(env, sac_trainer, logger, model_path=args.pretrained_path, manual_thresholds=args.manual_thresholds, n_test_runs=2)

    else:
        env.close()
        del env
        
        if args.pretrained_path != "":
            if args.load_buffer:
                replay_buffer.load(args.pretrained_path)
                replay_buffer.partial_delete(args.num_buffer_delete)
                replay_buffer.resize(args.buffer_size)
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
                                    # 'linear_lr_scheduler': [args.agent_lr, final_lr],
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
        
        replay_buffer.save(logger.save_path)
        
        env_kwargs['override_hyperparams']['reward_history_threshold'] = -10.0
        # test_pipeline(SelfTeachingBaseEnv(**env_kwargs), sac_trainer, logger, model_path=logger.save_path, manual_thresholds=args.manual_thresholds, all_samples=False, labeled_samples=False, conf_matrix=False, all_samples_labeled=False, trained_model=True)
        test_pipeline(SelfTeachingBaseEnv(**env_kwargs), sac_trainer, logger, model_path=logger.save_path, manual_thresholds=args.manual_thresholds)
