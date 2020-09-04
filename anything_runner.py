import os
from time import sleep

if __name__ == '__main__':
    
    commands = [# 'python3.7 main_basic.py --from-dataset ag_news --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-07-18_19-24-43_ag_news --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3', 
                # 'python3.7 main_basic.py --from-dataset ag_news --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-07-18_19-08-59_ag_news --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                # 'python3.7 main_basic.py --from-dataset cifar10 --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-18_13-48-24_cifar10 --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                # 'python3.7 main_basic.py --from-dataset cifar10 --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-21_21-14-09_cifar10 --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                # 'python3.7 main_basic.py --from-dataset cifar10_dense --dataset mnist_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-08_10-52-41_cifar10_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 6',
                # 'python3.7 main_basic.py --from-dataset cifar10_dense --dataset mnist_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-08_18-57-07_cifar10_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 6',
                
                # 'python3.7 main_basic.py --from-dataset cifar10_dense --dataset ag_news --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-09_04-54-43_cifar10_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                # 'python3.7 main_basic.py --from-dataset cifar10_dense --dataset ag_news --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-07_15-25-48_cifar10_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                # 'python3.7 main_basic.py --from-dataset mnist --dataset mnist_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-04_09-07-33_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                # 'python3.7 main_basic.py --from-dataset mnist --dataset mnist_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-01_19-18-19_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                # 'python3.7 main_basic.py --from-dataset mnist --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-05-24_21-17-32_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                # 'python3.7 main_basic.py --from-dataset mnist --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-05-27_21-15-20_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                'python3.7 main_basic.py --from-dataset mnist --dataset ag_news --pretrained-path /opt/workspace/host_storage_hdd/results/2020-06-01_19-18-19_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                'python3.7 main_basic.py --from-dataset mnist --dataset ag_news --pretrained-path /opt/workspace/host_storage_hdd/results/2020-05-31_08-20-02_mnist --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                
                'python3.7 main_basic.py --from-dataset mnist_dense --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-05-22_06-21-36_mnist_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                'python3.7 main_basic.py --from-dataset mnist_dense --dataset cifar10_dense --pretrained-path /opt/workspace/host_storage_hdd/results/2020-05-23_01-30-18_mnist_dense --load-model --learning-starts 10000 --n-warmup 0 --num-steps 70000 --num-workers 3',
                ]
    
    for command in commands:
        print("RUNNING:", command)
        os.system(command)
        
        sleep(5)