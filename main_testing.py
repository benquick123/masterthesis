import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist
from keras.utils import np_utils
import keras as k
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

from model import MNISTModel
from hidden_features import train_autoencoder, cluster_images


def plot(history, key='val_accuracy', name="tmp"):
    to_plot = [h[key] for h in history]
    
    plt.plot(to_plot)
    plt.draw()
    
    plt.savefig('/opt/workspace/host_storage_hdd/results/' + name + '.png')

def self_teaching(tau1, tau2):
    global X_train, y_train, X_unlabel, X_val, y_val, X_test, y_test
    
    y_unlabel_estimates = np.ones((X_unlabel.shape[0], y_train.shape[1])) * (1 / y_train.shape[1])
    estimated_lr = 0.3
    
    max_iter = 1000
    
    history = []
    
    for i in range(max_iter):
        print(i, "/", max_iter, end=" ")
        
        y_pred = model.predict(X_unlabel, batch_size=8192)
        
        y_unlabel_estimates = (1 - estimated_lr) * y_unlabel_estimates + estimated_lr * y_pred
        y_unlabel_estimates /= np.reshape(np.sum(y_unlabel_estimates, axis=1), (-1, 1))
        
        y_unlabel_estimates_argmax = np.argmax(y_unlabel_estimates, axis=1)
        y_unlabel_estimates_max = np.max(y_unlabel_estimates, axis=1)
        y_unlabel_estimates_indices = (y_unlabel_estimates_max > tau1) & (y_unlabel_estimates_max < tau2)
        print(np.sum(y_unlabel_estimates_indices), end="")
        
        y_pred_binary = np.zeros_like(y_pred)
        for j in range(len(y_unlabel_estimates_argmax)):
            y_pred_binary[j, y_unlabel_estimates_argmax[j]] = 1.0
        
        print(end="\r")
        
        history.append(model.fit(np.concatenate([X_train, X_unlabel[y_unlabel_estimates_indices]], axis=0), 
                                 np.concatenate([y_train, y_pred_binary[y_unlabel_estimates_indices]], axis=0), 
                                 validation_data=(X_val, y_val), epochs=1, batch_size=256, 
                                 verbose=1 if i % 25 == 0 else 0).history)
        
        """
            if i % 100 == 0:
                os.makedirs('/opt/workspace/host_storage_hdd/results/autoencoder_clustering/prob_dist_' + str(i), exist_ok=True)
                unique_groups = np.unique(X_unlabel_groups)
                
                for j, group in enumerate(unique_groups):
                    fig, ax = plt.subplots()
                    mask = group == X_unlabel_groups
                    y_plot_predictions = model.predict(X_unlabel[mask])
                    y_plot_predictions = np.mean(y_plot_predictions, axis=0)
                    
                    ax.bar(np.arange(len(y_plot_predictions)), y_plot_predictions)
                
                    plt.draw()
                    plt.savefig('/opt/workspace/host_storage_hdd/results/autoencoder_clustering/prob_dist_' + str(i) + '/group_' + str(j) + '.png')
                    
                    ax.cla()
                    fig.clf()
                    plt.close()
                    
                    fix, ax = plt.subplots(nrows=2, figsize=(6.4, 9.6))
                    
                    mask = group == X_val_groups
                    y_plot_predictions = model.predict(X_val[mask])
                    y_plot_predictions = np.mean(y_plot_predictions, axis=0)
                    ax[0].bar(np.arange(len(y_plot_predictions)), y_plot_predictions)
                    
                    ax[1].bar(np.arange(len(y_plot_predictions)), np.mean(y_val[mask], axis=0))
                    
                    plt.draw()
                    plt.savefig('/opt/workspace/host_storage_hdd/results/autoencoder_clustering/prob_dist_' + str(i) + '/group_' + str(j) + '_val.png')
                    
                    fig.clf()
                    plt.close()
        """
        
        """ 
            if i % 25 == 0:
                print("new figure")
                
                y_unlabel_estimates_max_argsort = np.argsort(y_unlabel_estimates_max)
                indices = np.linspace(0, X_unlabel.shape[0], 15*25, dtype="int", endpoint=False)

                y_unlabel_estimates_max_argsort = y_unlabel_estimates_max_argsort[indices]
                
                fig, axes = plt.subplots(15, 25, sharex=True, sharey=True, squeeze=True, figsize=(64, 48))
                for k, row in enumerate(axes):
                    for l, col in enumerate(row):
                        index = y_unlabel_estimates_max_argsort[k*15 + l]
                        col.imshow(X_unlabel[index].reshape((28, 28)))
                        col.set_title(str(k*15 + l) + " - " + "%3f" % (y_unlabel_estimates_max[index]))

                plt.draw()
                plt.savefig('/opt/workspace/host_storage_hdd/results/temporal_certainty_plots/x_unlabel_examples_' + str(i) + '.png')
        """
        
    # plot(history)
    
    return history


if __name__ == "__main__":
    
    generate_and_save = True
    n_clusters = 2000
    
    if generate_and_save:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # reshape, cast and scale in one step.
        X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
        X_train, X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
        
        X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
        
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        y_val = np_utils.to_categorical(y_val)
        
        autoencoder, encoder = train_autoencoder(X_train)
        groups, hidden_representations, group_centers = cluster_images(np.concatenate([X_unlabel, X_val], axis=0), encoder, n_clusters=n_clusters, plot=False)
        X_unlabel_groups, X_unlabel_hidden = groups[:len(X_unlabel)], hidden_representations[:len(X_unlabel)]
        X_val_groups, X_val_hidden = groups[len(X_unlabel):], hidden_representations[len(X_unlabel):]
        
        f = open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(n_clusters) + '.pickle', 'wb')
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_unlabel': X_unlabel,
            'X_unlabel_groups': X_unlabel_groups,
            'X_unlabel_hidden': X_unlabel_hidden,
            'X_val': X_val,
            'y_val': y_val,
            'X_val_groups': X_val_groups,
            'X_val_hidden': X_val_hidden,
            'X_test': X_test,
            'y_test': y_test,
            'groups_centers': group_centers,
        }, f)
        f.close()
    else:
        data = pickle.load(open('/opt/workspace/host_storage_hdd/mnist_preprocessed_' + str(n_clusters) + '.pickle', 'rb'))
        X_train = data['X_train']
        y_train = data['y_train']
        X_unlabel = data['X_unlabel']
        X_unlabel_groups = data['X_unlabel_groups']
        X_unlabel_hidden = data['X_unlabel_hidden']
        X_val = data['X_val']
        y_val = data['y_val']
        X_val_groups = data['X_val_groups']
        X_val_hidden = data['X_val_hidden']
        X_test = data['X_test']
        y_test = data['y_test']
        group_centers = data['group_centers']
    
    exit()
    print(X_train.shape, X_unlabel.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    
    model = MNISTModel(X_train.shape[1:], y_train.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    self_teaching(tau1=0.990, tau2=0.992)
    
    print(model.evaluate(X_test, y_test, batch_size=8192))
    
    """
        key = 'val_accuracy'
        
        history = []
        for i in range(10):
            print(i, "/ 10", end="\r")
            history.append(model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=256, verbose=0).history)
            model.reset()        
        print()

        to_plot = np.array([[h for h in run[key]] for run in history])
        to_plot_means = np.mean(to_plot, axis=0)
        to_plot_stds = np.std(to_plot, axis=0)
        
        plt.plot(to_plot_means, color="b", label="labeled only")
        plt.fill_between(np.arange(len(to_plot_means)), to_plot_means-to_plot_stds, to_plot_means+to_plot_stds, alpha=0.3, color="b")
        
        history = []
        for i in range(10):
            print("->", i, "/ 10")
            history.append(self_teaching(tau=0.98))
            model.reset()
            print()
        print()
        
        to_plot = np.array([[h[key] for h in run] for run in history]).squeeze(-1)
        to_plot_means = np.mean(to_plot, axis=0)
        to_plot_stds = np.std(to_plot, axis=0)
        
        plt.plot(to_plot_means, color="r", label="labeled + unlabeled (t=0.98)")
        plt.fill_between(np.arange(len(to_plot_means)), to_plot_means-to_plot_stds, to_plot_means+to_plot_stds, alpha=0.3, color="r")
        
        history = []
        for i in range(10):
            print("->", i, "/ 10")
            history.append(self_teaching(tau=0.0))
            model.reset()
            print()
        print()
        
        to_plot = np.array([[h[key] for h in run] for run in history]).squeeze(-1)
        to_plot_means = np.mean(to_plot, axis=0)
        to_plot_stds = np.std(to_plot, axis=0)
        
        plt.plot(to_plot_means, color="y", label="labeled + unlabeled (t=0.0)")
        plt.fill_between(np.arange(len(to_plot_means)), to_plot_means-to_plot_stds, to_plot_means+to_plot_stds, alpha=0.3, color="y")
        
        plt.legend()
        plt.draw()
        plt.savefig('/opt/workspace/host_storage_hdd/results/self_teach_comp.png')
    """
