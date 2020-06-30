import warnings
warnings.filterwarnings('ignore')

GPU_NUM = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
import errno

import time
import pickle

import torch

from env import SelfTeachingBaseEnv, accuracy_score

if __name__ == '__main__':
    
    env = SelfTeachingBaseEnv("cifar10", config_path="./config", override_hyperparams={"random_seed": 1})
    
    x_train, y_train = env.X_train, env.y_train
    x_unlabel, y_unlabel = env.X_unlabel, env.y_unlabel
    x_val, y_val = env.X_val, env.y_val
    x_test, y_test = env.X_test, env.y_test
    
    env.reset()
    
    model = env.model
    model_loss = env.model_loss
    model_optimizer = env.model_optimizer
    
    metrics = list()
    for epoch in range(300):
        print(epoch)
        epoch_metrics = dict()
        
        epoch_metrics["loop_start"] = time.time()
        
        model.fit(torch.cat((x_train, x_unlabel), axis=0), torch.cat((y_train, y_unlabel), axis=0), model_optimizer, model_loss, epochs=1)
        
        epoch_metrics["before_inference"] = time.time()
        
        predictions = []
        for i in range(1):
            predictions.append(model.predict(x_test, env.hyperparams['pred_batch_size']))
            if i == 0:
                epoch_metrics['1st inference end'] = time.time()
                
        print("accuracy:", accuracy_score(y_test, torch.argmax(predictions[-1], axis=1)))
        
        epoch_metrics["after_inference"] = time.time()
        
        epoch_metrics["predictions"] = predictions
        
        epoch_metrics["loop_end"] = time.time()
        metrics.append(epoch_metrics)
        
    pickle.dump(metrics, open("/opt/workspace/host_storage_hdd/regular_results.pkl", "wb"))
    