import math
import random

import numpy as np

import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


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

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_sizes=[64, 64], init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linears = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.linears.append(nn.Linear(state_dim + action_dim, hidden_layer_sizes[i]))
            else:
                self.linears.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
                
        self.linears.append(nn.Linear(hidden_layer_sizes[-1], 1))                

        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        self.linears[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # the dim 0 is number of samples  
        x = torch.cat([state, action], 1)
        
        for i in range(len(self.linears)):
            x = F.relu(self.linears[i](x))
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_sizes=[64, 64], action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linears = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.linears.append(nn.Linear(state_dim, hidden_layer_sizes[i]))
            else:
                self.linears.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
        
        self.mean_linear = nn.Linear(hidden_layer_sizes[-1], action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_layer_sizes[-1], action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.action_dim = action_dim

    def forward(self, state):
        x = F.relu(self.linears[0](state))
        
        for i in range(1, len(self.linears)):
            x = F.relu(self.linears[i](x))

        mean = (self.mean_linear(x))
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample()
        
        action_0 = torch.tanh(mean + std * z.cuda())  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        # THIS IS JUST ENTROPY!!!
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        # print(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample().cuda()
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        a = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()


class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        # initialized as [0.]: alpha->[1.]
        self.log_alpha=torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.log_alpha


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_layer_sizes=[64, 64], action_range=1., q_lr=3e-4, pi_lr=3e-4, alpha_lr=3e-4):
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_layer_sizes)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_layer_sizes, action_range)
        self.log_alpha = Alpha()

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        
        self.soft_q_optimizer1 = SharedAdam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = SharedAdam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=pi_lr)
        self.alpha_optimizer = SharedAdam(self.log_alpha.parameters(), lr=alpha_lr)

    def to_cuda(self):  # copy to specified gpu
        self.soft_q_net1 = self.soft_q_net1.cuda()
        self.soft_q_net2 = self.soft_q_net2.cuda()
        self.target_soft_q_net1 = self.target_soft_q_net1.cuda()
        self.target_soft_q_net2 = self.target_soft_q_net2.cuda()
        self.policy_net = self.policy_net.cuda()
        self.log_alpha = self.log_alpha.cuda()

    def update(self, batch_size, reward_scale=1., auto_entropy=True, gamma=0.99, soft_tau=0.003):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        
        # normalize rewards with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # Updating alpha wrt entropy
        # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            # it seems like log_prob needs to be as close to 1.0 as possible, for the loss to be small, i.e. the diff between log_prob and the other part must be as big as possible?
            # is this just basic maximization of entropy?
            alpha_loss = -(self.log_alpha() * (log_prob - 1.0 * self.action_dim).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action), self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
            
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1', map_location='cuda:0'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2', map_location='cuda:0'))
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location='cuda:0'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def worker(worker_id, sac_trainer, env_fn, env_kwargs, replay_buffer, num_steps, batch_size=64, learning_starts=100, n_updates=1, callback=None, AUTO_ENTROPY=True, DETERMINISTIC=False):
    with torch.cuda.device(worker_id % torch.cuda.device_count()):
        sac_trainer.to_cuda()
        
        rewards = []
        episode_reward = 0
        n_episodes = 0
        best_mean_train_rewards, best_mean_test_rewards = -np.inf, -np.inf
        
        env = env_fn(**env_kwargs)
        obs = env.reset()
        
        for step in range(num_steps):
            if callback is not None:
                callback(locals(), globals())
            
            if step < learning_starts:
                action = sac_trainer.policy_net.sample_action()
            else:
                action = sac_trainer.policy_net.get_action(obs, deterministic=DETERMINISTIC)
                
            next_obs, reward, done, _ = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            
            if done:
                obs = env.reset()
                rewards.append(episode_reward)
                episode_reward = 0
                n_episodes += 1
            else:
                obs = next_obs
                
            if replay_buffer.get_length() > batch_size:
                for i in range(n_updates):
                    sac_trainer.update(batch_size, auto_entropy=AUTO_ENTROPY)
                    
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
            

if __name__ == '__main__':
    pass
