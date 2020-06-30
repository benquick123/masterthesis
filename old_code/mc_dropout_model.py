import warnings
warnings.filterwarnings('ignore')

GPU_NUM = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
import errno

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif

from env import SelfTeachingBaseEnv

if __name__ == "__main__":
    env = SelfTeachingBaseEnv("cifar10", config_path="./config", override_hyperparams={"random_seed": 1})
    
    y_test = env.y_test
    
    mc_dropout = pickle.load(open("/opt/workspace/host_storage_hdd/mc_dropout_results.pkl", "rb"))
    regular = pickle.load(open("/opt/workspace/host_storage_hdd/regular_results.pkl", "rb"))
    
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    
    y_est = torch.ones((len(y_test), 10), device="cuda") * 0.1
    est_lr = 0.3
    
    for i in range(len(mc_dropout)):
        predictions = torch.stack(mc_dropout[i]['predictions']).exp()
        
        predictions_mean = predictions.mean(axis=0)
        predictions_std = predictions.std(axis=0)
        
        _x = 1 - predictions_std.gather(1, torch.argmax(predictions_mean, axis=1).view((-1, 1))).view(-1)
        _x = _x.cpu().numpy()
        x0.append(_x)
        
        _y = torch.argmax(predictions_mean, axis=1) == y_test
        _y = _y.cpu().numpy()
        y0.append(_y)
        
        predictions = regular[i]['predictions'][0].exp()
        
        y_est = (1 - est_lr) * y_est + est_lr * predictions
        
        _x = y_est.max(axis=1)[0]
        _x = _x.cpu().numpy()
        x1.append(_x)
        
        _y = torch.argmax(y_est, axis=1) == y_test
        _y = _y.cpu().numpy()
        y1.append(_y)
        
    x0 = np.concatenate(x0).reshape((-1, 1))
    x1 = np.concatenate(x1).reshape((-1, 1))
    y0 = np.concatenate(y0).reshape((-1, 1))
    y1 = np.concatenate(y1).reshape((-1, 1))
    
    mc = pd.DataFrame({'x': x0.reshape(-1), 'y': y0.reshape(-1)})
    mc['y'] = mc['y'].astype('category').cat.codes

    print("MC")
    print(mc.corr()['y'].loc['x'])
    
    reg = pd.DataFrame({'x': x1.reshape(-1), 'y': y1.reshape(-1)})
    reg['y'] = reg['y'].astype('category').cat.codes
    
    print("REG")
    print(reg.corr()['y'].loc['x'])
    
    print("X0:", mutual_info_classif(x0, y0))
    print("X1:", mutual_info_classif(x1, y1))