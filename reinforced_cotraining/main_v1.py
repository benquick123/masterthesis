import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

from preprocess import load_data

def cotraining_iter(clf1, clf2):
    pass


def cotraining(p=1, n=3, k=30, u=75):
    scores = []
    global X1_train, X2_train, y_train
    clf1 = RandomForestClassifier(n_estimators=100) # GaussianNB() # MultinomialNB()
    clf2 = RandomForestClassifier(n_estimators=100) # GaussianNB() # MultinomialNB()
    
    # still need seperate y for both classifiers?
    unlabeled_permutated_indices = np.arange(X1_unlabeled.shape[0])
    unlabeled_pool_indices = unlabeled_permutated_indices[:u]
    unlabeled_permutated_indices = unlabeled_permutated_indices[u:]
    
    for iteration in range(k):
        print(iteration, unlabeled_pool_indices.shape, unlabeled_permutated_indices.shape, X1_train.shape, X2_train.shape, y_train.shape, end=", ")
        clf1.fit(X1_train, y_train)
        clf2.fit(X2_train, y_train)
        
        tmp_X1 = X1_unlabeled[unlabeled_pool_indices]
        tmp_X2 = X2_unlabeled[unlabeled_pool_indices]
        
        y1_pred = clf1.predict_proba(tmp_X1)
        y1_argsort_neg = np.argsort(y1_pred[:, 0])[::-1][:n]
        y1_argsort_pos = np.argsort(y1_pred[:, 1])[::-1][:p]
        y1_pred_binary = np.argmax(y1_pred, axis=1)
        
        y2_pred = clf2.predict_proba(tmp_X2)
        y2_argsort_neg = np.argsort(y2_pred[:, 0])[::-1][:n]
        y2_argsort_pos = np.argsort(y2_pred[:, 1])[::-1][:p]
        y2_pred_binary = np.argmax(y2_pred, axis=1)
        
        clf1_indices = np.array(np.append(y1_argsort_neg, y1_argsort_pos), dtype="int")
        clf2_indices = np.array(np.append(y2_argsort_neg, y2_argsort_pos), dtype="int")
        
        clf2_indices = np.setdiff1d(clf2_indices, clf1_indices)
        if len(clf1_indices) + len(clf2_indices) != 2*p + 2*n:
            print("DIFF", clf1_indices, clf2_indices, end="")
        
        new_indices = np.append(clf1_indices, clf2_indices) 
        
        X1_train = np.append(X1_train, tmp_X1[new_indices], axis=0)
        X2_train = np.append(X2_train, tmp_X2[new_indices], axis=0)
        y_train = np.append(y_train, y1_pred_binary[clf1_indices])
        y_train = np.append(y_train, y2_pred_binary[clf2_indices])
        
        unlabeled_pool_indices = np.delete(unlabeled_pool_indices, new_indices) # ? is new_indices ok?
        
        unlabeled_pool_indices = np.append(unlabeled_pool_indices, unlabeled_permutated_indices[:len(new_indices)])
        unlabeled_permutated_indices = unlabeled_permutated_indices[len(new_indices):]
        
        scores.append([accuracy_score(y_test, clf1.predict(X1_test)), accuracy_score(y_test, clf2.predict(X2_test))])
        
        print()
        if len(unlabeled_pool_indices) == 0:
            break
        
    return np.transpose(np.array(scores))
        

if __name__ == "__main__":
    X1_train, X2_train, y_train, X1_unlabeled, X2_unlabeled, X1_test, X2_test, y_test = load_data(encode=False)
    print(X1_train.shape, X2_train.shape, y_train.shape)
    print(X1_unlabeled.shape, X2_unlabeled.shape)
    print(X1_test.shape, X2_test.shape, y_test.shape)
    
    scores = cotraining(n=2, p=2, u=776, k=30)
    
    for timeseries, label in zip(range(scores.shape[0]), ["clf1", "clf2"]):
        plt.plot(scores[timeseries, :], label=label)
    plt.plot(np.mean(scores, axis=0), label="mean")
    plt.legend()
    plt.savefig("/opt/workspace/host_storage_hdd/results/tmp.png")
    
    