import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
from keras.optimizers import Adam

from preprocess import load_data, load_data_v2
from model import SimpleModel, LSTMTextModel

def cotraining_iter(clf1, clf2):
    pass


def cotraining(p=1, n=3, k=30, u=75, n_epoch=1):
    scores = []
    global X1_train, X2_train, y_train
    # clf1 = RandomForestClassifier(n_estimators=100) # GaussianNB() # MultinomialNB()
    # clf2 = RandomForestClassifier(n_estimators=100) # GaussianNB() # MultinomialNB()
    # clf1 = SimpleModel(X1_train.shape[1:], y_train.shape[1], layer_dims=[X1_train.shape[1] // 100, X1_train.shape[1] // 200])
    # clf1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # clf2 = SimpleModel(X2_train.shape[1:], y_train.shape[1], layer_dims=[X2_train.shape[1] // 5, X2_train.shape[1] // 10])
    # clf2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    clf1 = LSTMTextModel(X1_train.shape[1:], y_train.shape[1])
    clf1.compile(optimizer=Adam(lr=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
    
    clf2 = LSTMTextModel(X1_train.shape[1:], y_train.shape[1])
    clf2.compile(optimizer=Adam(lr=0.0005 ), loss="categorical_crossentropy", metrics=["accuracy"])
    
    unlabeled_permutated_indices = np.random.permutation(np.arange(X1_unlabeled.shape[0]))
    unlabeled_pool_indices = unlabeled_permutated_indices[:u]
    unlabeled_permutated_indices = unlabeled_permutated_indices[u:]
    
    clf1.fit(X1_train, y_train, epochs=1, verbose=0)
    clf2.fit(X2_train, y_train, epochs=1, verbose=0)
    
    for iteration in range(k):
        print(iteration, unlabeled_pool_indices.shape, unlabeled_permutated_indices.shape, X1_train.shape, X2_train.shape, y_train.shape, end=" ")
        clf1.fit(X1_train, y_train, epochs=n_epoch, verbose=0)
        clf2.fit(X2_train, y_train, epochs=n_epoch, verbose=0)
        
        tmp_X1 = X1_unlabeled[unlabeled_pool_indices]
        tmp_X2 = X2_unlabeled[unlabeled_pool_indices]
        
        y1_pred = clf1.predict(tmp_X1)
        y1_argsort_neg = np.argsort(y1_pred[:, 0])[::-1][:n]
        y1_argsort_pos = np.argsort(y1_pred[:, 1])[::-1][:p]
        y1_pred_binary = np.round(y1_pred)
        
        y2_pred = clf2.predict(tmp_X2)
        y2_argsort_neg = np.argsort(y2_pred[:, 0])[::-1][:n]
        y2_argsort_pos = np.argsort(y2_pred[:, 1])[::-1][:p]
        y2_pred_binary = np.round(y2_pred)
        
        clf1_indices = np.array(np.append(y1_argsort_neg, y1_argsort_pos), dtype="int")
        clf2_indices = np.array(np.append(y2_argsort_neg, y2_argsort_pos), dtype="int")
        
        clf2_indices = np.setdiff1d(clf2_indices, clf1_indices)
        if len(clf1_indices) + len(clf2_indices) != 2*p + 2*n:
            print("DIFF", clf1_indices, clf2_indices, end=" ")
        
        new_indices = np.append(clf1_indices, clf2_indices) 
        
        X1_train = np.append(X1_train, tmp_X1[new_indices], axis=0)
        X2_train = np.append(X2_train, tmp_X2[new_indices], axis=0)
        y_train = np.append(y_train, y1_pred_binary[clf1_indices], axis=0)
        y_train = np.append(y_train, y2_pred_binary[clf2_indices], axis=0)
        
        unlabeled_pool_indices = np.delete(unlabeled_pool_indices, new_indices)
        
        unlabeled_pool_indices = np.append(unlabeled_pool_indices, unlabeled_permutated_indices[:len(new_indices)])
        unlabeled_permutated_indices = unlabeled_permutated_indices[len(new_indices):]
        
        scores.append([accuracy_score(np.argmax(y_test, axis=1), np.argmax(clf1.predict(X1_test), axis=1)), accuracy_score(np.argmax(y_test, axis=1), np.argmax(clf2.predict(X2_test), axis=1))])
        
        print(scores[-1])
        if len(unlabeled_pool_indices) == 0:
            break
        
    return np.transpose(np.array(scores))
        

if __name__ == "__main__":
    X1_train, X2_train, y_train, X1_unlabeled, X2_unlabeled, X1_test, X2_test, y_test = load_data_v2()
    print(X1_train.shape, X2_train.shape, y_train.shape)
    print(X1_unlabeled.shape, X2_unlabeled.shape)
    print(X1_test.shape, X2_test.shape, y_test.shape)

    scores = cotraining(n=2, p=2, u=10000, k=30)
    
    for timeseries, label in zip(range(scores.shape[0]), ["clf1", "clf2"]):
        plt.plot(scores[timeseries, :], label=label)
    plt.plot(np.mean(scores, axis=0), label="mean")
    plt.legend()
    plt.savefig("/opt/workspace/host_storage_hdd/results/tmp.png")
    
    