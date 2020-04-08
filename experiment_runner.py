import os
import time


if __name__ == '__main__':
    args = [
        # ("ag_news", 9),
        # ("cifar10", 6),
        ("dbpedia", 6),
        ("fasion_mnist", 18),
        ("imdb", 15),
        ("mnist", 18),
        ("svhn", 12),
        ("usps", 18)
    ]
    
    for dataset, num_workers in args:
        run_str = "python3.7 main_basic.py --dataset " + dataset + " --num-workers " + str(num_workers)
        print("RUNNING:", run_str)
        code = os.system(run_str)
        print(dataset, "-", num_workers, "- DONE")
        
        time.sleep(5)
    
