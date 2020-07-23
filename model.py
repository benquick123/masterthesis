import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch import Tensor
from torch.nn.init import calculate_gain
from torch.nn.init import kaiming_uniform_ as kaiming_uniform_init
from torch.nn.init import xavier_uniform_ as xavier_uniform_init
from torch.nn.init import zeros_ as zeros_init
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from datasets import CustomImageDataset
from utils import ECELoss


class Autoencoder(nn.Module):
    
    def __init__(self, input_size, encoder_hidden_layer_sizes, decoder_hidden_layer_sizes):
        super(Autoencoder, self).__init__()
        
        assert encoder_hidden_layer_sizes[-1] == decoder_hidden_layer_sizes[0], "Last encoder and first decoder layer sizes must be the same!"
        
        layers = []
        for i in range(len(encoder_hidden_layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, encoder_hidden_layer_sizes[i]))
            else:
                layers.append(nn.Linear(encoder_hidden_layer_sizes[i-1], encoder_hidden_layer_sizes[i]))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        for i in range(1, len(decoder_hidden_layer_sizes)):
            layers.append(nn.Linear(decoder_hidden_layer_sizes[i-1], decoder_hidden_layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(decoder_hidden_layer_sizes[-1], input_size))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)
            
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def fit(self, x, y, optimizer, loss_fn, epochs=10, batch_size=64, verbose=1, gpu_id=0):
        x = Tensor(x)
        y = Tensor(y)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if verbose > 0:
                print("EPOCH: %3d/%3d - loss: %.3f" % (epoch, epochs, loss.cpu().detach().numpy()))


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
    
    def copy_weights(self, old_model):
        assert all([self.linears[i].weight.shape == old_model.linears[i].weight.shape and self.linears[i].bias.shape == old_model.linears[i].bias.shape for i in range(len(self.linears))])
        
        for i in range(len(self.linears)):
            self.linears[i].weight.data = old_model.linears[i].weight.data
            self.linears[i].bias.data = old_model.linears[i].bias.data
            
    def freeze(self, freeze_mask):
        raise NotImplementedError


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
        
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action):
        # the dim 0 is number of samples  
        x = torch.cat([state, action], 1)
        
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
            
        x = self.linears[-1](x)
        return x
    
    def copy_weights(self, old_model, tile_action_tensors=True, noise_weight=0.0, old_model_weight=1.0):
        if not tile_action_tensors:
            copy_start_index = 0
        else:
            copy_start_index = 1
            assert self.state_dim == old_model.state_dim and self.action_dim % old_model.action_dim == 0
            
            n_repeats = self.action_dim // old_model.action_dim
            self.linears[0].weight.data = torch.cat([old_model.linears[0].weight.data[:, :self.state_dim], 
                                                     self.linears[0].weight.data[:, self.state_dim:] * noise_weight + old_model.linears[0].weight.data[:, self.state_dim:].repeat(1, n_repeats) * old_model_weight], axis=1)
            self.linears[0].bias.data = torch.cat([old_model.linears[0].bias.data[:self.state_dim], 
                                                   self.linears[0].bias.data[self.state_dim:] * noise_weight + old_model.linears[0].bias.data[self.state_dim:] * old_model_weight], axis=0)
        
        assert all([self.linears[i].weight.shape == old_model.linears[i].weight.shape and self.linears[i].bias.shape == old_model.linears[i].bias.shape for i in range(copy_start_index, len(self.linears))])
        for i in range(copy_start_index, len(self.linears)):
            self.linears[i].weight.data = old_model.linears[i].weight.data
            self.linears[i].bias.data = old_model.linears[i].bias.data
            
    def freeze(self, freeze_mask):
        for i, freeze_bool in enumerate(freeze_mask[::-1]):
            self.linears[i].weight.requires_grad = not freeze_bool
            self.linears[i].bias.requires_grad = not freeze_bool


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
        
        # squashing correction; samples will have different probabilities after applying tanh to them.
        log_pi = (log_pi - torch.log(1.0 - stoh_action ** 2 + epsilon)).sum(dim=1, keepdim=True)
        return det_action, stoh_action, log_pi

    def get_action(self, state, deterministic):
        state = state.unsqueeze(0).view((-1, self.state_dim)).cuda()

        mean, log_std = self.forward(state)
        if deterministic:
            action = self.action_range * torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(0, 1)
            z = normal.sample(sample_shape=mean.size()).cuda()
            action = self.action_range * torch.tanh(mean + std * z)
            
        return action[0]

    def copy_weights(self, old_model, tile_action_tensors=True, noise_weight=0.0, old_model_weight=1.0):        
        assert all([self.linears[i].weight.shape == old_model.linears[i].weight.shape and self.linears[i].bias.shape == old_model.linears[i].bias.shape for i in range(len(self.linears))])
        for i in range(len(self.linears)):
            self.linears[i].weight.data = old_model.linears[i].weight.data
            self.linears[i].bias.data = old_model.linears[i].bias.data
            
        if not tile_action_tensors:
            assert all([o.weight.shape == n.weight.shape and o.bias.shape == n.bias.shape for o, n in [(old_model.mean_linear, self.mean_linear), (old_model.log_std_linear, self.log_std_linear)]])
            self.mean_linear.weight.data = old_model.mean_linear.weight.data
            self.mean_linear.bias.data = old_model.mean_linear.bias.data
            self.log_std_linear.weight.data = old_model.log_std_linear.weight.data
            self.log_std_linear.bias.data = old_model.log_std_linear.bias.data
        else:
            assert all([all([n_shape % o_shape == 0 for o_shape, n_shape in zip(o.weight.shape, n.weight.shape)]) for o, n in [(old_model.mean_linear, self.mean_linear), (old_model.log_std_linear, self.log_std_linear)]])
            assert all([all([n_shape % o_shape == 0 for o_shape, n_shape in zip(o.bias.shape, n.bias.shape)]) for o, n in [(old_model.mean_linear, self.mean_linear), (old_model.log_std_linear, self.log_std_linear)]])
            
            n_repeats = self.mean_linear.weight.shape[0] // old_model.mean_linear.weight.shape[0]
            
            self.mean_linear.weight.data = self.mean_linear.weight.data * noise_weight + old_model.mean_linear.weight.data.repeat(n_repeats, 1) * old_model_weight
            self.mean_linear.bias.data = self.mean_linear.bias.data * noise_weight + old_model.mean_linear.bias.data.repeat(n_repeats) * old_model_weight
            
            self.log_std_linear.weight.data = self.log_std_linear.weight.data * noise_weight + old_model.log_std_linear.weight.data.repeat(n_repeats, 1) * old_model_weight
            self.log_std_linear.bias.data = self.log_std_linear.bias.data * noise_weight + old_model.log_std_linear.bias.data.repeat(n_repeats) * old_model_weight
            
    def freeze(self, freeze_mask):
        for i, freeze_bool in enumerate(freeze_mask):
            if i < len(self.linears):
                self.linears[i].weight.requires_grad = not freeze_bool
                self.linears[i].bias.requires_grad = not freeze_bool
            else:
                self.mean_linear.weight.requires_grad = not freeze_bool
                self.mean_linear.bias.requires_grad = not freeze_bool
                self.log_std_linear.weight.requires_grad = not freeze_bool
                self.log_std_linear.bias.requires_grad = not freeze_bool


class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        # initialized as [0.]: alpha->[1.]
        self.log_alpha=torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.log_alpha
    
    def copy_weights(self, old_model, noise=0.0):
        self.log_alpha = old_model.log_alpha


class Temperature(nn.Module):
    
    def __init__(self):
        super(Temperature, self).__init__()
        self.temperature = torch.tensor(torch.ones(1), requires_grad=True, device="cuda")
        # self.temperature = nn.Parameter(torch.ones(1))
        self.ece_loss = ECELoss()
        self.nll_loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        self.optimizer = torch.optim.Adam([self.temperature], lr=0.01)

    def forward(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, logits, labels, verbose=1, gpu_id=0):
        if verbose == 1:
            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = self.nll_loss(logits.detach(), labels).item()
            before_temperature_ece = self.ece_loss(logits.detach(), labels).item()
            
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
            
        # Next: optimize the temperature w.r.t. NLL
        loss = self.nll_loss(self.forward(logits.detach()), labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if verbose == 1:
            # Calculate NLL and ECE after temperature scaling
            with torch.no_grad():
                after_temperature_nll = self.nll_loss(self.forward(logits.detach()), labels).item()
                after_temperature_ece = self.ece_loss(self.forward(logits.detach()), labels).item()
            
            print('Optimal temperature: %.3f' % self.temperature.item())
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))


class CustomModel(nn.Module):
    
    def __init__(self):
        super(CustomModel, self).__init__()
    
    def fit(self, x, y, optimizer, loss_fn, batch_size=64, epochs=10, verbose=1, gpu_id=0):
        # expects 'x' and 'y' to be tensors and not DataLoaders (as opposed to self.predict())
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(x) % batch_size == 1) # , pin_memory=True)
        
        for epoch in range(epochs):
            for x_batch, y_batch in data_loader:
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if verbose > 0:
                print("EPOCH: %3d/%3d - loss: %.3f" % (epoch, epochs, loss.cpu().detach().numpy()))
                
    def fit_semi(self, x_label, y_label, x_unlabel, y_unlabel, optimizer, loss_fn, batch_size=64, epochs=10, verbose=1, gpu_id=0, alpha=1):
        if x_unlabel is None or len(x_unlabel) == 0 or alpha < 1e-6:
            self.fit(x_label, y_label, optimizer, loss_fn, batch_size, epochs, verbose, gpu_id)
        else:
            labeled_dataset = TensorDataset(x_label, y_label)
            unlabeled_dataset = TensorDataset(x_unlabel, y_unlabel)
            
            labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
            
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            
            n_batches = max([len(x_label), len(x_unlabel)]) // batch_size
            
            for epoch in range(epochs):
                for batch in range(n_batches):
                    try:
                        x_batch_label, y_batch_label = next(labeled_iter)
                    except StopIteration:
                        labeled_iter = iter(labeled_loader)
                        x_batch_label, y_batch_label = next(labeled_iter)
                        
                    try:
                        x_batch_unlabel, y_batch_unlabel = next(unlabeled_iter)
                    except StopIteration:
                        x_batch_unlabel, y_batch_unlabel = None, None
                            
                    y_pred_label = self.forward(x_batch_label)
                    if x_batch_unlabel is not None and y_batch_unlabel is not None:
                        y_pred_unlabel = self.forward(x_batch_unlabel)
                        loss = loss_fn(y_pred_label, y_batch_label) + alpha * loss_fn(y_pred_unlabel, y_batch_unlabel)
                    else:
                        loss = loss_fn(y_pred_label, y_batch_label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if verbose > 0:
                    print("EPOCH: %3d/%3d - loss: %.3f - alpha: %.3f" % (epoch, epochs, loss.cpu().detach().numpy(), alpha))
                
    def predict(self, x, batch_size=8192, gpu_id=0, eval_mode=True):
        # expects 'x' to either be a tensor that fits into memory or batch_size has to be set to appropriate size.
        if eval_mode:
            self.eval()
            
        with torch.no_grad():
            if batch_size != -1:
                y = None
                for i in range(len(x) // batch_size):
                    y_pred = F.log_softmax(self.forward(x[i * batch_size:(i+1) * batch_size]))
                    y = y_pred if y is None else torch.cat([y, y_pred], axis=0)
                
                remaining_samples = len(x) % batch_size
                y_pred = F.log_softmax(self.forward(x[-remaining_samples:]))
                y = y_pred if y is None else torch.cat([y, y_pred], axis=0)
            else:
                y = F.log_softmax(self.forward(x))

        if eval_mode:
            self.train()
        return y.detach()
    
    def reset(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                kaiming_uniform_init(layer.weight)
            elif isinstance(layer, nn.Conv2d):
                xavier_uniform_init(layer.weight)
            
            if (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)) and layer.bias is not None:
                zeros_init(layer.bias)
                
    def forward(self, x):
        return self.layers(x)
        

class DenseModel(CustomModel):
    
    def __init__(self, input_dim, output_dim, layer_sizes=[256, 128]):
        super(DenseModel, self).__init__()
        self.layer_sizes = layer_sizes
        self._create_model(input_dim, output_dim)
        
    def _create_model(self, input_dim, output_dim):
        layers = []
        for i in range(len(self.layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_dim[0], self.layer_sizes[i]))
            else:
                layers.append(nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(self.layer_sizes[-1], output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    
class ConvModel(CustomModel):
    
    def __init__(self, input_dim, output_dim, n_filters=[32, 64, 128], fully_connected_layer=128):
        super(ConvModel, self).__init__()
        self.n_filters = n_filters
        self.fully_connected_layer = fully_connected_layer
        self._create_model(input_dim, output_dim)
        
    def _create_model(self, input_dim, output_dim):
        min_dropout = 0.2
        max_dropout = 0.5
        dropout_range = torch.arange(min_dropout, max_dropout, (max_dropout - min_dropout + 0.099)/(len(self.n_filters) + 1))
        
        layers = []
        for i in range(len(self.n_filters)):
            if i == 0:
                layers.append(nn.Conv2d(input_dim[0], self.n_filters[i], kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(self.n_filters[i-1], self.n_filters[i], kernel_size=3, padding=1))
                layers.append(nn.Dropout2d())
                
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # layers.append(nn.BatchNorm2d(self.n_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            
            # layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i], kernel_size=3, padding=1))
            # layers.append(nn.BatchNorm2d(self.n_filters[i]))
            # layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(dropout_range[i]))
            # 
            # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        # layers.append(nn.Dropout2d(dropout_range[-1]))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.n_filters[-1] * (input_dim[1] // 2 ** len(self.n_filters)) * (input_dim[2] // 2 ** len(self.n_filters)), self.fully_connected_layer))
        # layers.append(nn.BatchNorm1d(self.fully_connected_layer, eps=0.001))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_range[-1]))
        layers.append(nn.Linear(self.fully_connected_layer, output_dim))
        
        self.layers = nn.Sequential(*layers)


class TextModel(CustomModel):
    
    def __init__(self, input_dim, output_dim, emb_dim, embedding_path=None, dense_layer_sizes=[128]):
        super(TextModel, self).__init__()
        self.dense_layer_sizes = dense_layer_sizes
        self.embedding_path = embedding_path
        self._create_model(input_dim, emb_dim, output_dim)
    
    def _create_model(self, input_dim, emb_dim, output_dim):
        vectors = pickle.load(open(self.embedding_path, "rb"))
        kernel_divisor = 1
        
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.embedding.weight.requires_grad = False
        
        self.pooling = nn.AvgPool2d((input_dim[0] // kernel_divisor, 1), stride=None)
        
        linear_layers = []
        for i in range(len(self.dense_layer_sizes)):
            if i == 0:
                linear_layers.append(nn.Linear(kernel_divisor * emb_dim, self.dense_layer_sizes[i]))
            else:
                linear_layers.append(nn.Linear(self.dense_layer_sizes[i-1], self.dense_layer_sizes[i]))
            linear_layers.append(nn.ReLU())
            
        linear_layers.append(nn.Linear(self.dense_layer_sizes[-1], output_dim))
        
        self.layers = [self.embedding, self.pooling] + linear_layers
        self.linear_layers = nn.ModuleList(linear_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.pooling(x)
        # x = x.squeeze(1)
        
        x = x.view((x.shape[0], -1))
        
        for layer in self.linear_layers:
            x = layer(x)
            
        return x
