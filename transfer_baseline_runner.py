import os
import time
import re
import argparse

import numpy as np
import pandas as pd

def analyse():
    # build up relevant directories and parse them
    parent_dir = '/opt/workspace/host_storage_hdd/results'
    p = re.compile('2020.*_to_.*transfer_baseline_test')
    dirs = [s for s in os.listdir(parent_dir) if p.match(s)]
    
    data = []
    
    for d in dirs:
        from_dataset, dataset = "_".join(d.split('_')[2:-3]).split("_to_")
        
        log_file = open(os.path.join(parent_dir, d, "log.log"), "r")
        log = log_file.readlines()
        try:
            acc = float(log[-1].split(": ")[-1])
        except ValueError:
            print("ValueError:", d)
            continue
        
        row = [from_dataset, dataset, acc]
        data.append(row)
        
    data = np.array(data)
    df = pd.DataFrame(data, columns=['from', 'to', "acc"])
    df['from'] = df['from'].astype(str)
    df['to'] = df['to'].astype(str)
    df['acc'] = df['acc'].astype('float')

    # print(df)

    results = df.groupby(['from', 'to'])['acc'].agg(["mean", "std", 'count'])
    print(results)        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", action="store_true", help="Whether to show just statistics.")
    args = parser.parse_args()
    
    if args.analyse:
        analyse()
    else:
        datasets = ['mnist', 'mnist_dense', 'cifar10', 'cifar10_dense', 'ag_news']
        
        pretrained_paths = ['/opt/workspace/host_storage_hdd/results/2020-05-21_20-25-28_mnist_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-05-22_06-21-36_mnist_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-05-22_16-16-18_mnist_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-05-23_01-30-18_mnist_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-05-23_11-06-27_mnist_dense',
                            
                            '/opt/workspace/host_storage_hdd/results/2020-05-24_21-17-32_mnist',
                            '/opt/workspace/host_storage_hdd/results/2020-05-27_21-15-20_mnist',
                            '/opt/workspace/host_storage_hdd/results/2020-05-31_08-20-02_mnist',
                            '/opt/workspace/host_storage_hdd/results/2020-06-01_19-18-19_mnist',
                            '/opt/workspace/host_storage_hdd/results/2020-06-04_09-07-33_mnist',
                            
                            '/opt/workspace/host_storage_hdd/results/2020-06-07_15-25-48_cifar10_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-06-08_01-18-13_cifar10_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-06-08_10-52-41_cifar10_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-06-08_18-57-07_cifar10_dense',
                            '/opt/workspace/host_storage_hdd/results/2020-06-09_04-54-43_cifar10_dense',
                            
                            '/opt/workspace/host_storage_hdd/results/2020-06-18_13-48-24_cifar10',
                            '/opt/workspace/host_storage_hdd/results/2020-06-21_21-14-09_cifar10',
                            
                            '/opt/workspace/host_storage_hdd/results/2020-07-17_22-10-03_ag_news',
                            '/opt/workspace/host_storage_hdd/results/2020-07-17_22-13-10_ag_news',
                            '/opt/workspace/host_storage_hdd/results/2020-07-18_19-08-59_ag_news',
                            '/opt/workspace/host_storage_hdd/results/2020-07-18_19-24-43_ag_news',
                            '/opt/workspace/host_storage_hdd/results/2020-07-18_19-26-23_ag_news']
        
        path_postfix = 'transfer_baseline'
        
        for i, pretrained_path in enumerate(pretrained_paths):
            from_dataset = "_".join(pretrained_path.split("_")[4:])
            folder_name = pretrained_path.split('/')[-1]
            parent_folder = "/".join(pretrained_path.split('/')[:-1])
            
            for dataset in datasets:
                if dataset != from_dataset:
                    new_folder_name = folder_name + "_to_" + dataset + "_" + path_postfix + "_test"
                    if new_folder_name not in set(os.listdir(parent_folder)):
                        run_str = "python3.7 main_basic.py --test --from-dataset %s --dataset %s --pretrained-path %s --new-folder --path-postfix %s" % (from_dataset, dataset, pretrained_path, path_postfix)
                        print("RUNNING %d: %s" % (i, run_str))
                        os.system(run_str)
                        
                        time.sleep(5)
                    else:
                        print("SKIPPING %d: %s" % (i, new_folder_name))