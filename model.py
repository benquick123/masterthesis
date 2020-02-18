import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.init import kaiming_uniform_ as kaiming_uniform_init, zeros_ as zeros_init, xavier_uniform_ as xavier_uniform_init, calculate_gain
from torch import Tensor

import numpy as np


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
    
    def fit(self, x, y, optimizer, loss_fn, epochs=10, batch_size=64, verbose=1):
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


class MNISTModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(MNISTModel, self).__init__()
        self._create_model(input_dim, output_dim)
        self.n_linear_layers = np.sum([isinstance(layer, nn.Linear) for layer in self.layers])        
        
    def _create_model(self, input_dim, output_dim):        
        layers = []
        
        layers.append(nn.Linear(input_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(256, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(128, output_dim))
        # layers.append(nn.Softmax())
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    def reset(self):
        layer_counter = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer_counter += 1
                if layer_counter < self.n_linear_layers:
                    kaiming_uniform_init(layer.weight)
                else:
                    xavier_uniform_init(layer.weight, gain=calculate_gain('sigmoid'))
                if layer.bias is not None:
                    zeros_init(layer.bias)
        
    def fit(self, x, y, optimizer, loss_fn, batch_size=64, epochs=10, verbose=1, gpu_id=0):
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch
                _, y_batch = y_batch.max(1)
                
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if verbose > 0:
                print("EPOCH: %3d/%3d - loss: %.3f" % (epoch, epochs, loss.cpu().detach().numpy()))
                
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y = F.log_softmax(self.forward(x))
        self.train()
        return y.detach()
