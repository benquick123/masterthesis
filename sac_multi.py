import math
import random
import traceback
import os

import numpy as np
import pickle

import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import ValueNetwork, PolicyNetwork, SoftQNetwork, Alpha


class SharedAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SharedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                
                # how exactly does this work?
                # this is the only thing added compared to original implementation.
                device = p.device
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, hash_str=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, hash_str)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, _ = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
    
    def save_buffer(self, filepath):
        pickle.dump((self.buffer, self.position), open(os.path.join(filepath, "replay_buffer.pkl"), "wb"))
        
    def load_buffer(self, filepath):
        self.buffer, self.position = pickle.load(open(os.path.join(filepath, "replay_buffer.pkl"), "rb"))


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_layer_sizes=[64, 64], action_range=1., q_lr=3e-4, pi_lr=3e-4, alpha_lr=3e-4, v_lr=3e-4):
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim

        self.q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        self.q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        
        self.value_net = ValueNetwork(state_dim, hidden_layer_sizes)
        self.target_value_net = ValueNetwork(state_dim, hidden_layer_sizes)
        
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_layer_sizes, action_range)
        self.log_alpha = Alpha()
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.q_criterion_1 = nn.MSELoss()
        self.q_criterion_2 = nn.MSELoss()
        self.v_criterion = nn.MSELoss()
        
        self.q_optimizer_1 = SharedAdam(self.q_net_1.parameters(), lr=q_lr)
        self.q_optimizer_2 = SharedAdam(self.q_net_2.parameters(), lr=q_lr)
        self.v_optimizer = SharedAdam(self.value_net.parameters(), lr=v_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=pi_lr)
        self.alpha_optimizer = SharedAdam(self.log_alpha.parameters(), lr=alpha_lr)

    def to_cuda(self):  # copy to specified gpu
        self.q_net_1 = self.q_net_1.cuda()
        self.q_net_2 = self.q_net_2.cuda()
        self.value_net = self.value_net.cuda()
        self.target_value_net = self.target_value_net.cuda()
        
        self.policy_net = self.policy_net.cuda()
        self.log_alpha = self.log_alpha.cuda()

    def update(self, batch_size, gamma=0.99, soft_tau=0.003):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()
        
        self.alpha_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.q_optimizer_1.zero_grad()
        self.q_optimizer_2.zero_grad()
        self.v_optimizer.zero_grad()

        # get stoh_action, det_action and log_pi
        det_action, stoh_action, log_pi = self.policy_net.evaluate(state)
        
        # compute new alpha and backprop it
        a_loss = -(self.log_alpha() * (log_pi - self.action_dim).detach()).mean()
        a_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha().exp()
            
        with torch.no_grad():
            target_v_next = self.target_value_net(next_state)
            q_backup = reward + gamma * (1 - done) * target_v_next
        
        q_loss_1 = self.q_criterion_1(self.q_net_1(state, action), q_backup.detach())
        q_loss_1.backward()
        self.q_optimizer_1.step()
        
        q_loss_2 = self.q_criterion_2(self.q_net_2(state, action), q_backup.detach())
        q_loss_2.backward()
        self.q_optimizer_2.step()
        
        q_1_act = self.q_net_1(state, stoh_action)
        q_2_act = self.q_net_1(state, stoh_action)
        
        p_loss = (self.alpha * log_pi - q_1_act).mean()
        p_loss.backward()
        self.policy_optimizer.step()
        
        with torch.no_grad():
            v_backup = torch.min(q_1_act, q_2_act) - self.alpha * log_pi
        v_loss = self.v_criterion(self.value_net(state), v_backup.detach())
        v_loss.backward()
        self.v_optimizer.step()
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        
        return a_loss, q_loss_1, q_loss_2, v_loss, p_loss
    
    def update_lr(self, new_lr):
        for optimizer in [self.q_optimizer_1, self.q_optimizer_2, self.v_optimizer, self.policy_optimizer, self.alpha_optimizer]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
    def save_model(self, path):
        torch.save(self.q_net_1.state_dict(), path + '_q1')
        torch.save(self.q_net_2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.value_net.state_dict(), path + '_value')
        torch.save(self.target_value_net.state_dict(), path + '_target_value')

    def load_model(self, path, evaluation=False, gpu_id=0):
        self.q_net_1.load_state_dict(torch.load(path + '_q1'))
        self.q_net_2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.value_net.load_state_dict(torch.load(path + '_value'))
        self.target_value_net.load_state_dict(torch.load(path + '_target_value'))

        if evaluation:
            self.q_net_1.eval()
            self.q_net_2.eval()
            self.policy_net.eval()
            self.value_net.eval()
            self.target_value_net.eval()


def worker(worker_id, sac_trainer, env_fn, env_kwargs, replay_buffer, num_steps, batch_size=64, learning_starts=100, n_warmup=100, n_updates=1, linear_lr_scheduler=None, callback=None, callback_kwargs={}, log_path=None, DETERMINISTIC=False):
    if log_path is not None:
        writer = SummaryWriter(log_dir=log_path + "WORKER_" + str(worker_id))
        writer.add_scalar("Rewards/episodeReward", 0, 0)
        writer.add_scalar('Actions/meanTestActions', 0, 0)
        writer.add_scalar('Accuracies/testAccuracies', 0, 0)
        writer.add_scalar('Accuracies/trainAccuracies', 0, 0)
        writer.add_scalar('Misc/numSelectedSamples', 0, 0)
    else:
        writer = None
    
    if isinstance(linear_lr_scheduler, list):
        learning_rate_changer = lambda x : (linear_lr_scheduler[0] - linear_lr_scheduler[1]) * (1 - x / num_steps) + linear_lr_scheduler[1]
    
    with torch.cuda.device(worker_id % torch.cuda.device_count()):
        sac_trainer.to_cuda()
        
        rewards = []
        episode_reward = 0
        n_episodes = 0
        n_episode_steps = 0
        best_mean_train_rewards, best_mean_test_rewards = -np.inf, -np.inf
        
        env_kwargs['override_hyperparams']['worker_id'] = worker_id
        env = env_fn(**env_kwargs)
        obs = env.reset()
        
        for step in range(num_steps):
            if callback is not None:
                callback(locals(), globals(), **callback_kwargs)
            
            if step < n_warmup:
                action = torch.Tensor(env.action_space.sample()).cuda()
            else:
                action = sac_trainer.policy_net.get_action(obs, deterministic=DETERMINISTIC)
                
            next_obs, reward, done, info = env.step(action)
            hash_str = str((n_episodes + 1) * worker_id)
            
            replay_buffer.push(obs.cpu().detach().numpy(), action.cpu().detach().numpy(), reward.cpu().detach().numpy(), next_obs.cpu().detach().numpy(), done, hash_str)
            
            episode_reward += reward
            n_episode_steps += 1
            
            if done:
                obs = env.reset()
                rewards.append(episode_reward.cpu().detach().numpy())
                if writer is not None:
                    writer.add_scalar('Rewards/episodeReward', episode_reward, step)
                    writer.add_scalar('Accuracies/trainAccuracies', info['val_acc'], step)
                
                episode_reward = 0
                n_episode_steps = 0
                n_episodes += 1
            else:
                obs = next_obs
                
            if replay_buffer.get_length() > batch_size and step >= learning_starts:
                for i in range(n_updates):
                    alpha_loss, q_value_loss1, q_value_loss2, value_loss, policy_loss = sac_trainer.update(batch_size)
            else:
                alpha_loss, q_value_loss1, q_value_loss2, value_loss, policy_loss = 0, 0, 0, 0, 0
            
            if linear_lr_scheduler is not None:
                sac_trainer.update_lr(learning_rate_changer(step))
            
            if writer is not None:
                writer.add_scalar('Rewards/reward', reward, step)
                writer.add_scalar('Actions/meanTrainActions', ((action + 1) / 2).view((-1, 2))[:, 0].mean(), step)
                writer.add_scalar("Losses/alphaLoss", alpha_loss, step)
                writer.add_scalar('Losses/qValueLoss1', q_value_loss1, step)
                writer.add_scalar('Losses/qValueLoss2', q_value_loss2, step)
                writer.add_scalar('Losses/policyLoss', policy_loss, step)
                writer.add_scalar('Losses/valueLoss', value_loss, step)
                writer.add_scalar('Misc/replayBufferSize', replay_buffer.get_length(), step)
                writer.add_scalar('Misc/numSelectedSamples', info['num_samples'].cpu().numpy(), step)
                
        if worker_id == 0:
            sac_trainer.save_model(log_path + 'final_sac_self_teaching')
                    
        return sac_trainer


def share_parameters(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()
            
