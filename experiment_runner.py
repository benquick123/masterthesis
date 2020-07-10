import os
import time


if __name__ == '__main__':
    args = [
        ("ag_news", 9),
        ("ag_news", 9),
        ("ag_news", 9),
        ("ag_news", 9),
        ("ag_news", 9)
    ]
    
    for i, (dataset, num_workers) in enumerate(args):
        run_str = "python3.7 main_basic.py --dataset " + dataset + " --num-workers " + str(num_workers)
        print("RUN #%d:" % (i), run_str)
        code = os.system(run_str)
        print(dataset, "-", num_workers, "- DONE")
        
        time.sleep(5)
