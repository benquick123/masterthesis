import math
import random
import traceback

import numpy as np

import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


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


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layer_sizes=[64, 64], init_w=3e-3):
        super(ValueNetwork, self).__init__()


        self.linears = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.linears.append(nn.Linear(state_dim, hidden_layer_sizes[i]))
            else:
                self.linears.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
        
        self.linears.append(nn.Linear(hidden_layer_sizes[-1], 1))
        
        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        self.linears[-1].bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linears[0](state))
        
        for i in range(1, len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[-1](x)
            
        return x


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
        
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
            
        x = self.linears[-1](x)
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
        self.state_dim = state_dim

    def forward(self, state):
        # traceback.print_stack()
        # print(state.size(), "\n")
        x = F.relu(self.linears[0](state))
        
        for i in range(1, len(self.linears)):
            x = F.relu(self.linears[i](x))

        mean = self.mean_linear(x)
        # mean = (self.mean_linear(x))
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(sample_shape=mean.size())
        
        # reparametrization trick
        stoh_action = mean + std * z.cuda()
        
        # log_pi
        log_pi = Normal(mean, std).log_prob(stoh_action)
        
        stoh_action = self.action_range * torch.tanh(stoh_action)
        det_action = self.action_range * torch.tanh(mean)
        log_pi = (log_pi - torch.log(1.0 - stoh_action ** 2 + epsilon)).sum(dim=1, keepdim=True)
        return mean, stoh_action, log_pi

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).cuda().view((-1, self.state_dim))

        mean, log_std = self.forward(state)
        if deterministic:
            action = self.action_range * torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(0, 1)
            z = normal.sample(sample_shape=mean.size()).cuda()
            action = self.action_range * torch.tanh(mean + std * z)
            
        return action.cpu().detach().numpy()


class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        # initialized as [0.]: alpha->[1.]
        self.log_alpha=torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.log_alpha


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

    def update(self, batch_size, auto_entropy=True, gamma=0.99, soft_tau=0.003):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()
        
        # get stoh_action, det_action and log_pi
        det_action, stoh_action, log_pi = self.policy_net.evaluate(state)
        
        # compute new alpha and backprop
        if auto_entropy:
            a_loss = -(self.log_alpha() * (log_pi - self.action_dim).detach()).mean()
            self.alpha_optimizer.zero_grad()
            a_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha().exp()
        else:
            self.alpha = 1.0
            a_loss = 0.0
            
        target_v_next = self.value_net(next_state)
        q_1 = self.q_net_1(state, stoh_action.detach())
        q_2 = self.q_net_2(state, stoh_action.detach())
        
        q_backup = reward + gamma * (1 - done) * target_v_next
        q_loss_1 = self.q_criterion_1(q_1, q_backup.detach())
        self.q_optimizer_1.zero_grad()
        q_loss_1.backward()
        self.q_optimizer_1.step()
        q_loss_2 = self.q_criterion_2(q_2, q_backup.detach())
        self.q_optimizer_2.zero_grad()
        q_loss_2.backward()
        self.q_optimizer_2.step()
        
        v_backup = torch.min(q_1, q_2) - self.alpha * log_pi
        v_loss = self.v_criterion(self.value_net(state), v_backup.detach())
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        p_loss = (self.alpha * log_pi - q_1.detach()).mean()
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        
        return a_loss, q_loss_1, q_loss_2, v_loss, p_loss

    def save_model(self, path):
        torch.save(self.q_net_1.state_dict(), path + '_q1')
        torch.save(self.q_net_2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch_save(self.value_net.state_dict(), path + '_value')
        torch_save(self.target_value_net.state_dict(), path + '_target_value')
        

    def load_model(self, path):
        self.q_net_1.load_state_dict(torch.load(path + '_q1', map_location='cuda:0'))
        self.q_net_2.load_state_dict(torch.load(path + '_q2', map_location='cuda:0'))
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location='cuda:0'))
        self.value_net.load_state_dict(torch.load(path + '_value', map_location='cuda:0'))
        self.target_value_net.load_state_dict(torch.load(path + '_target_value', map_location='cuda:0'))

        self.q_net_1.eval()
        self.q_net_2.eval()
        self.policy_net.eval()
        self.value_net.eval()
        self.target_value_net.eval()


def worker(worker_id, sac_trainer, env_fn, env_kwargs, replay_buffer, num_steps, batch_size=64, learning_starts=100, n_updates=1, callback=None, log_path=None, AUTO_ENTROPY=True, DETERMINISTIC=False):
    if log_path is not None:
        writer = SummaryWriter(log_dir=log_path + "WORKER_" + str(worker_id))
        writer.add_scalar("Rewards/episode_reward", 0, 0)
        writer.add_scalar('Actions/meanTestActions', 0, 0)
        writer.add_scalar('Accuracies/testAccuracies', 0, 0)
    else:
        writer = None
    
    with torch.cuda.device(worker_id % torch.cuda.device_count()):
        sac_trainer.to_cuda()
        
        rewards = []
        episode_reward = 0
        n_episodes = 0
        n_episode_steps = 0
        best_mean_train_rewards, best_mean_test_rewards = -np.inf, -np.inf
        
        # env_kwargs['ID'] = worker_id
        env = env_fn(**env_kwargs)
        obs = env.reset()
        
        for step in range(num_steps):
            if callback is not None:
                callback(locals(), globals())
            
            if step < learning_starts:
                action = env.action_space.sample()
            else:
                action = sac_trainer.policy_net.get_action(obs, deterministic=DETERMINISTIC)
                
            next_obs, reward, done, _ = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            n_episode_steps += 1
            
            if done:
                obs = env.reset()
                rewards.append(episode_reward)
                if writer is not None:
                    writer.add_scalar('Rewards/episode_reward', episode_reward, step)
                
                episode_reward = 0
                n_episode_steps = 0
                n_episodes += 1
            else:
                obs = next_obs
                
            if replay_buffer.get_length() > batch_size and step >= learning_starts:
                alpha_loss, q_value_loss1, q_value_loss2, value_loss, policy_loss = sac_trainer.update(batch_size, auto_entropy=AUTO_ENTROPY)
            else:
                alpha_loss, q_value_loss1, q_value_loss2, value_loss, policy_loss = 0, 0, 0, 0, 0
            
            if writer is not None:
                writer.add_scalar('Rewards/reward', reward, step)
                writer.add_scalar('Actions/meanTrainActions', np.mean(action), step)
                writer.add_scalar("Losses/alphaLoss", alpha_loss, step)
                writer.add_scalar('Losses/qValueLoss1', q_value_loss1, step)
                writer.add_scalar('Losses/qValueLoss2', q_value_loss2, step)
                writer.add_scalar('Losses/policyLoss', policy_loss, step)
                writer.add_scalar('Losses/valueLoss', value_loss, step)
                writer.add_scalar('Misc/replayBufferSize', replay_buffer.get_length(), step)
                
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
            

if __name__ == '__main__':
    pass
