import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.init import kaiming_uniform_ as kaiming_uniform_init, zeros_ as zeros_init, xavier_uniform_ as xavier_uniform_init, calculate_gain
from torch import Tensor

from datasets import CustomImageDataset


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


class CustomModel(nn.Module):
    
    def __init__(self):
        super(CustomModel, self).__init__()
    
    def fit(self, x, y, optimizer, loss_fn, batch_size=64, epochs=10, verbose=1, gpu_id=0, transforms=None):
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(x) % batch_size == 1)
        
        for epoch in range(epochs):
            for x_batch, y_batch in data_loader:
                # x_batch = x_batch.cuda(gpu_id)
                # y_batch = y_batch.cuda(gpu_id)
                # _, y_batch = y_batch.max(1)
                
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if verbose > 0:
                print("EPOCH: %3d/%3d - loss: %.3f" % (epoch, epochs, loss.cpu().detach().numpy()))
                
    def predict(self, x, batch_size=64, gpu_id=0):
        if batch_size == -1:
            batch_size = len(x)
            
        self.eval()
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        y = None
        
        with torch.no_grad():
            for [x_batch] in data_loader:
                y_pred = F.log_softmax(self.forward(x_batch))
                if y is None:
                    y = torch.tensor(y_pred)
                else:
                    y = torch.cat([y, y_pred], axis=0)
        # with torch.no_grad():
        #     y = F.softmax(self.forward(x))
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
        print(self.layers(x))
        exit()
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
        dropout_range = torch.range(min_dropout, max_dropout, (max_dropout - min_dropout + 0.099)/(len(self.n_filters) + 1))
        
        layers = []
        for i in range(len(self.n_filters)):
            if i == 0:
                layers.append(nn.Conv2d(input_dim[0], self.n_filters[i], kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(self.n_filters[i-1], self.n_filters[i], kernel_size=3, padding=1))
                
            layers.append(nn.BatchNorm2d(self.n_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(self.n_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_range[i]))
            
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.n_filters[-1] * (input_dim[1] // 2 ** len(self.n_filters)) * (input_dim[2] // 2 ** len(self.n_filters)), self.fully_connected_layer))
        layers.append(nn.BatchNorm1d(self.fully_connected_layer, eps=0.001))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_range[-1]))
        layers.append(nn.Linear(self.fully_connected_layer, output_dim))
        
        self.layers = nn.Sequential(*layers)


class TextModel(CustomModel):
    
    def __init__(self, input_dim, output_dim, emb_dim, lstm_cells=128, dense_layer_sizes=[128]):
        super(TextModel, self).__init__()
        self.lstm_cells = lstm_cells
        self.dense_layer_sizes = dense_layer_sizes
    
    def _create_model(self, input_dim, emb_dim, output_dim):
        
        layers = []
        layers.append(nn.Embedding(input_dim, emb_dim))
        """
        layers.append(nn.LSTM(emb_dim, self.lstm_cells))
        
        for i in range(self.dense_layer_sizes):
            layers.append(nn.Linear)
        """
        self.layers = nn.Sequential(*layers)
