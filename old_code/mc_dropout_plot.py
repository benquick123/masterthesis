import warnings
warnings.filterwarnings('ignore')

GPU_NUM = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
import errno

import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import SelfTeachingBaseEnv, accuracy_score


def plot_accuracies(mc, reg):
    plt.plot(np.arange(len(mc)), mc, label="MC")
    plt.plot(np.arange(len(reg)), reg, label="REG")
    
    plt.legend()
    plt.draw()
    plt.savefig("/opt/workspace/host_storage_hdd/mc_vs_reg_acc.svg")


def plot_time(mc, reg):
    X = np.arange(len(mc))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(mc)
    print(reg)
    ax.bar(X, mc, width=0.25)
    ax.bar(X+0.25, reg, width=0.25)
    
    plt.draw()
    plt.savefig("/opt/workspace/host_storage_hdd/mc_vs_reg_time.svg")
    

def plot_stds(mc_corr, mc_max, reg_corr=None, reg_max=None, avg_corr=None, avg_max=None):
    plt.plot(np.arange(len(mc_corr)), mc_corr, label="MC corr")
    plt.plot(np.arange(len(mc_max)), mc_max, label="MC max")
    if reg_corr is not None:
        plt.plot(np.arange(len(reg_corr)), reg_corr, label="REG corr")
        plt.plot(np.arange(len(reg_max)), reg_max, label="REG max")
        
    if avg_corr is not None:
        plt.plot(np.arange(len(avg_corr)), avg_corr, label="AVG corr")
        plt.plot(np.arange(len(avg_max)), avg_max, label="AVG max")
    
    plt.legend()
    plt.ylabel("std")
    plt.xlabel("epoch #")
    plt.draw()
    plt.savefig("/opt/workspace/host_storage_hdd/mc_vs_reg_stds_0_3.svg")
    
    
def plot_sample(mc_corr, mc_max):
    plt.plot(mc_corr, label="MC corr")
    plt.plot(mc_max, label="MC max")
    
    plt.legend()
    plt.draw()
    plt.savefig("/opt/workspace/host_storage_hdd/mc_vs_reg_sample.svg")


if __name__ == '__main__':
    env = SelfTeachingBaseEnv("cifar10", config_path="./config", override_hyperparams={"random_seed": 1})
    
    x_train, y_train = env.X_train, env.y_train
    x_unlabel, y_unlabel = env.X_unlabel, env.y_unlabel
    x_val, y_val = env.X_val, env.y_val
    x_test, y_test = env.X_test, env.y_test
    
    mc_dropout = pickle.load(open("/opt/workspace/host_storage_hdd/mc_dropout_results.pkl", "rb"))
    regular = pickle.load(open("/opt/workspace/host_storage_hdd/regular_results.pkl", "rb"))
    
    # TIME
    # mc_before_inference = np.mean([epoch_data['before_inference'] - epoch_data['loop_start'] for epoch_data in mc_dropout])
    # mc_st_inference = np.mean([epoch_data['1st inference end'] - epoch_data['loop_start'] for epoch_data in mc_dropout])
    # mc_after_inference_end = np.mean([epoch_data['after_inference'] - epoch_data['loop_start'] for epoch_data in mc_dropout])
    # mc_acc = [mc_before_inference, mc_st_inference, mc_after_inference_end]
    # 
    # reg_before_inference = np.mean([epoch_data['before_inference'] - epoch_data['loop_start'] for epoch_data in regular])
    # reg_st_inference = np.mean([epoch_data['1st inference end'] - epoch_data['loop_start'] for epoch_data in regular])
    # reg_after_inference_end = np.mean([epoch_data['after_inference'] - epoch_data['loop_start'] for epoch_data in regular])
    # reg_acc = [reg_before_inference, reg_st_inference, reg_after_inference_end]
    # 
    # plot_time(mc_acc, reg_acc)
    
    # ACCURACIES
    # mc_predictions = []
    # mc_accuracies = []
    # reg_accuracies = []
    # for i in range(len(mc_dropout)):
    #     ith_predictions = torch.stack(mc_dropout[i]["predictions"])
    #     ith_predictions = ith_predictions.mean(axis=0)
    #     mc_predictions.append(ith_predictions)
    #     
    #     mc_accuracies.append(accuracy_score(y_test, torch.argmax(ith_predictions, axis=1)))
    #     
    #     reg_accuracies.append(accuracy_score(y_test, torch.argmax(torch.stack(regular[i]["predictions"]).mean(axis=0), axis=1)))
    #     
    # mc_predictions = torch.stack(mc_predictions)
    # 
    # mc_accuracies = torch.stack(mc_accuracies).cpu()
    # reg_accuracies = torch.stack(reg_accuracies).cpu()
    # 
    # plot_accuracies(mc_accuracies, reg_accuracies)
    
    # CONFIDENCES: STDs
    # mc_stds_corr = []
    # mc_stds_max = []
    # reg_stds_corr = []
    # reg_stds_max = []
    # 
    # y_estimates = torch.ones((len(y_test), 10), device="cuda") * 0.1
    # estimates_lr = 0.3
    # averaged_stds_corr = []
    # averaged_stds_max = []
    # 
    # for i in range(len(mc_dropout)):
    #     ith_predictions = torch.stack(mc_dropout[i]['predictions'])
    #     ith_predictions_std = ith_predictions.std(axis=0)
    #     ith_predictions_mean = ith_predictions.mean(axis=0)
    #     
    #     mc_stds_corr.append(ith_predictions_std.gather(1, y_test.view((-1, 1))).view(-1).mean())
    #     mc_stds_max.append(ith_predictions_std.gather(1, torch.argmax(ith_predictions_mean, axis=1).view((-1, 1))).view(-1).mean())
    #     
    #     ith_predictions = torch.stack(regular[i]['predictions'])[0]
    # 
    #     ith_predictions_corr = ith_predictions.gather(1, y_test.view((-1, 1))).view(-1).mean()
    #     ith_predictions_max = torch.max(ith_predictions, axis=1)[0].view(-1).mean()
    #     
    #     reg_stds_corr.append(ith_predictions_corr)
    #     reg_stds_max.append(ith_predictions_max)
    #     
    #     y_estimates = (1 - estimates_lr) * y_estimates + estimates_lr * ith_predictions.exp()
    #     averaged_stds_corr.append(y_estimates.gather(1, y_test.view((-1, 1))).view(-1).mean())
    #     averaged_stds_max.append(torch.max(y_estimates, axis=1)[0].view(-1).mean())
    #     
    #     
    # plot_stds(mc_stds_corr, mc_stds_max, reg_stds_corr, reg_stds_max, averaged_stds_corr, averaged_stds_max)
    
    # ONE SAMPLE ANALYSIS
    # mc_sample_corr = []
    # mc_sample_max = []
    # sample_i = 98
    # for i in range(len(mc_dropout)):
    #     sample_predictions = torch.stack(mc_dropout[i]['predictions']).exp()[:, sample_i]
    #     sample_predictions_mean = sample_predictions.mean(axis=0).cpu()
    #     sample_predictions_std = sample_predictions.std(axis=0).cpu()
    #     
    #     if i == len(mc_dropout)-1:
    #         print("pred == true?", (torch.argmax(sample_predictions_mean) == y_test[sample_i]).item())
    #     
    #     mc_sample_corr.append(sample_predictions_std[y_test[sample_i]].item())
    #     mc_sample_max.append(sample_predictions_std[torch.argmax(sample_predictions_mean)].item())
    # 
    # plot_sample(mc_sample_corr, mc_sample_max)