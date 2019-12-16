import h5py
from tensorflow import keras as k
from sklearn.preprocessing import LabelBinarizer
import numpy as np
np.random.seed(42)

import tensorflow as tf

class SVHN:

    @staticmethod
    def load_data(val_size=0, path="data/"):
        with h5py.File(path + "SVHN_train.hdf5", "r") as f:
            shape = f["X"].shape
            x_train = f["X"][:shape[0]-val_size]
            y_train = f["Y"][:shape[0]-val_size].flatten()
            x_val = f["X"][shape[0]-val_size:]
            y_val = f["Y"][shape[0]-val_size:].flatten()

        with h5py.File(path + "SVHN_test.hdf5", "r") as f:
            x_test = f["X"][:]
            y_test = f["Y"][:].flatten()

        y_train = k.utils.to_categorical(y_train, 10)
        y_test = k.utils.to_categorical(y_test, 10)
        
        if val_size > 0:
            y_val = k.utils.to_categorical(y_val, 10)
            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        else:
            return (x_train, y_train), (x_test, y_test)


def preprocess(x_train, y_train, labeled_size, unlabeled_size, valid_size=0.0, epsilon=0.05, n_classes=None):
    if isinstance(labeled_size, float):
        labeled_size = int(len(x_train) * labeled_size)
    if isinstance(unlabeled_size, float):
        unlabeled_size = int(len(x_train) * unlabeled_size)
    if valid_size > 0 and isinstance(valid_size, float):
        valid_size = int(len(x_train * valid_size))
        
    if n_classes is None:
        n_classes = y_train.shape[1]

    indices_perm = np.arange(len(y_train))
    np.random.shuffle(indices_perm)
    x_train = x_train[indices_perm]
    y_train = y_train[indices_perm]

    x_train_labeled = []
    y_train_labeled = []
    x_valid = []
    y_valid = []
    x_train_unlabeled = []
    y_train_unlabeled = []

    n_samples_per_label = int(labeled_size / n_classes)
    n_samples_per_unlabel = int(unlabeled_size / n_classes)
    n_samples_per_valid = int(valid_size / n_classes)
    for label in range(n_classes):        
        label_indices = np.where(y_train[:, label] == 1)[0]
        choice_indices = label_indices[np.random.permutation(label_indices.shape[0])]

        label_indices = choice_indices[:n_samples_per_label]
        x_train_labeled.append(x_train[label_indices])
        y_train_labeled.append(y_train[label_indices])

        choice_indices = choice_indices[n_samples_per_label:]
        unlabel_indices = choice_indices[:n_samples_per_unlabel]
        x_train_unlabeled.append(x_train[unlabel_indices])
        y_train_unlabeled.append(y_train[unlabel_indices])
        
        if valid_size > 0:
            choice_indices = choice_indices[n_samples_per_unlabel:]
            valid_indices = choice_indices[:n_samples_per_valid]
            x_valid.append(x_train[valid_indices])
            y_valid.append(y_train[valid_indices])

    x_train_labeled = np.vstack(x_train_labeled)
    y_train_labeled = np.vstack(y_train_labeled)
    x_train_unlabeled = np.vstack(x_train_unlabeled)
    y_train_unlabeled = np.vstack(y_train_unlabeled)
    
    label_indices = np.random.permutation(y_train_labeled.shape[0])
    x_train_labeled = x_train_labeled[label_indices]
    y_train_labeled = y_train_labeled[label_indices, :n_classes]
    
    unlabel_indices = np.random.permutation(y_train_unlabeled.shape[0])
    x_train_unlabeled = x_train_unlabeled[unlabel_indices]
    y_train_unlabeled = y_train_unlabeled[unlabel_indices, :n_classes]
    
    if valid_size > 0:
        x_valid = np.vstack(x_valid)
        y_valid = np.vstack(y_valid)
        valid_indices = np.random.permutation(y_valid.shape[0])
        x_valid = x_valid[valid_indices]
        y_valid = y_valid[valid_indices, :n_classes]

    # initialize y_estimated variable
    """y_train_estimated = np.random.randint(0, y_train_unlabeled.shape[1], size=y_train_unlabeled.shape[0])
    y_train_estimated = LabelBinarizer().fit_transform(y_train_estimated)
    y_train_estimated = simplex_projection(tf.Variable(y_train_estimated, dtype=tf.float32, trainable=True), 64)"""
    """y_train_estimated = np.random.uniform(size=y_train_unlabeled.shape)
    y_train_estimated = y_train_estimated / np.reshape(np.sum(y_train_estimated, axis=1), (-1, 1))"""
    # y_train_estimated = tf.Variable(y_train_estimated, dtype=tf.float32, trainable=True)
    """y_train_estimated = np.random.normal(0.1, 0.01, size=y_train_unlabeled.shape)
    y_train_estimated = y_train_estimated / np.reshape(np.sum(y_train_estimated, axis=1), (-1, 1))"""

    return (x_train_labeled, y_train_labeled), (x_train_unlabeled, y_train_unlabeled), (x_valid, y_valid)
