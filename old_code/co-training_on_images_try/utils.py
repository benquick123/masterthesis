import tensorflow as tf
from tensorflow import keras as k
import numpy as np

import os
import shutil
import pickle


def generate_points(dim, delta):
    def sums(length, total_sum):
        if length == 1:
            yield (total_sum,)
        else:
            for value in range(total_sum + 1):
                for permutation in sums(length - 1, total_sum - value):
                    yield (value,) + permutation
    
    sum_to = int(1 / (dim * delta))
    return np.array(list(sums(dim, dim * sum_to))) / (dim * sum_to)


def get_cluster_centers(dim, delta, n_clusters, seed=None):
    np.random.seed(seed)
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-10) / np.log(len(p)))

    cluster_centers = generate_points(dim, delta)
    clusters = []
    blacklist = set()
    
    entropies = []
    for point in cluster_centers:
        entropies.append(entropy(point))
    entropies = np.clip(entropies, 0.0, 1.0)

    indices = np.ones(len(entropies), dtype=bool)
    for threshold in np.arange(0, 1, 1/n_clusters):
        new_indices = (entropies >= threshold) == (entropies < threshold + 1/n_clusters)
        new_indices = np.arange(len(entropies))[new_indices]
        if len(new_indices) > 0:
            indices = new_indices
        
        if len(indices) > 0:
            while True:
                random_index = np.random.randint(0, len(indices))
                if indices[random_index] in blacklist:
                    indices = np.delete(indices, random_index)
                else:
                    break
            clusters.append(cluster_centers[indices[random_index]])
            
            blacklist.add(indices[random_index])
            indices = np.delete(indices, random_index)
        else:
            while True:
                random_index = np.random.randint(0, len(entropies))
                if random_index not in blacklist:
                    blacklist.add(random_index)
                    break
            clusters.append(cluster_centers[random_index])

    np.random.seed(None)
    return np.array(clusters)


def rampup(rampup_period):
    def _rampup(epoch):
        if epoch == 0:
            return 0.0
        elif epoch < rampup_period:
            phase = 1.0 - epoch / rampup_period
            return tf.exp(-5.0 * phase * phase)
        else:
            return 1.0
    
    return _rampup


def rampdown(rampdown_period, n_epoch):
    def _rampdown(epoch):
        if epoch < n_epoch - rampdown_period:
            return 1.0
        else:
            phase = 1.0 - (n_epoch - epoch) / rampdown_period
            return tf.exp(-12.0 * phase * phase)
    
    return _rampdown


def data_generator(x, y, training=True):
    i = 0
    num_samples = len(y)
    while True:
        i = i % num_samples
        yield x[i], y[i], i
        i += 1
        

def get_image_augmentations(x_label, x_unlabel=None, BATCH_SIZE=50):
    # if this is slow, put ImageDataGenerator back to dataset_generator.
    data_generator = k.preprocessing.image.ImageDataGenerator(
        horizontal_flip=False,
        width_shift_range=4,
        height_shift_range=4,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    
    x_label_aug_student = next(data_generator.flow(x_label, batch_size=BATCH_SIZE, shuffle=False))
    x_label_aug_teacher = next(data_generator.flow(x_label, batch_size=BATCH_SIZE, shuffle=False))
    
    if x_unlabel is not None:
        x_unlabel_aug_student = next(data_generator.flow(x_unlabel, batch_size=BATCH_SIZE, shuffle=False))
        x_unlabel_aug_teacher = next(data_generator.flow(x_unlabel, batch_size=BATCH_SIZE, shuffle=False))
    else:
        x_unlabel_aug_student, x_unlabel_aug_teacher = None, None
    
    return x_label_aug_student, x_label_aug_teacher, x_unlabel_aug_student, x_unlabel_aug_teacher
        

class Log:
    current_time = 0
    log = False

    @staticmethod
    def check_folders(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print("Created folder:", path)

    @staticmethod
    def _print(*s, **kwargs):
        if Log.log:
            Log.check_folders("logs/" + Log.current_time)
            
            f = open("logs/" + Log.current_time + "/log.txt", "a")
            print(*s, **kwargs, file=f)
            f.close()

        print(*s, **kwargs)

    @staticmethod
    def save_self():
        print("SAVING SELF.")
        Log.check_folders("logs/" + Log.current_time)
        
        filenames = ["main.py", "model.py", "preprocess.py", "utils.py", "reinforcement.py"]
        for filename in filenames: 
            shutil.copyfile(filename, "logs/" + Log.current_time + "/" + filename)

    @staticmethod
    def save_object(obj, filename):
        Log.check_folders("logs/" + Log.current_time)
        f = open("logs/" + Log.current_time + "/" + filename, "wb")
        pickle.dump(obj, f)
        f.close()
