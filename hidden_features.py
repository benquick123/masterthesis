from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from torch.optim import Adam
from torch.nn import MSELoss
from torch import Tensor

from model import Autoencoder


def train_autoencoder(x):
    autoencoder = Autoencoder(np.prod(x.shape[1:]).astype(np.int32), [128, 64, 32], [32, 64, 128]).cuda()
    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    loss = MSELoss()
    autoencoder.fit(x, x, optimizer, loss, epochs=100, batch_size=32, verbose=0)
    
    return autoencoder, autoencoder.encoder
    

def cluster_images(x, encoder, n_clusters=80, plot=False):
    kmeans = KMeans(n_clusters=n_clusters)
    
    x = Tensor(x).cuda()
    x_hidden_features = encoder(x).cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    
    groups = kmeans.fit_predict(x_hidden_features)

    if plot:
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            mask = groups == group
            group_members = x[mask]
            
            fig, ax = plt.subplots(8, 8, sharex=True, sharey=True)
            for row in ax:
                for col in row:
                    index = np.random.randint(0, len(group_members))
                    col.imshow(group_members[index].reshape((28, 28)))
            plt.draw()
            plt.savefig('/opt/workspace/host_storage_hdd/results/autoencoder_clustering/group_' + str(group) + '_' + str(len(group_members)) + '.png')
            
    return groups, x_hidden_features, kmeans.cluster_centers_
    

if __name__ == "__main__":
    pass
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape, cast and scale in one step.
    X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
    X_train, X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
    
    X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
    
    autoencoder, encoder = train_autoencoder(X_train)
    
    cluster_images(np.concatenate([X_val, X_unlabel], axis=0), encoder)
    """
    
    