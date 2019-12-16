import keras as k
from keras.datasets import mnist
from keras.layers import Input, Dense
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def train_autoencoder(x):
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    
    hidden_features = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(hidden_features)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    
    autoencoder = k.models.Model(input_img, decoded)    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, batch_size=32, epochs=100, verbose=0)
    
    encoder = k.models.Model(autoencoder.input, autoencoder.layers[3].output)
    
    return autoencoder, encoder
    
    # fig, ax = plt.subplots(10, 2, sharex=True, sharey=True)
    # for i, row in enumerate(ax):
    #     row[0].imshow(x[i].reshape((28, 28)), cmap='gray')
    #     row[1].imshow(autoencoder.predict(x[i:i+1]).reshape((28, 28)), cmap='gray')
    #     print(encoder.predict(x[i:i+1]))
        
    # plt.draw()
    # plt.savefig('/opt/workspace/host_storage_hdd/results/autoencoder.png')
    

def cluster_images(x, encoder, n_clusters=80, plot=False):
    kmeans = KMeans(n_clusters=n_clusters)
    x_hidden_features = encoder.predict(x)    
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
            
    return groups
    

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape, cast and scale in one step.
    X_train = X_train.reshape(X_train.shape[:1] + (np.prod(X_train.shape[1:]), )).astype('float32') / 255
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.025)
    X_train, X_unlabel, y_train, _ = train_test_split(X_train, y_train, test_size=0.99)
    
    X_test = X_test.reshape(X_test.shape[:1] + (np.prod(X_test.shape[1:]), )).astype('float32') / 255
    
    autoencoder, encoder = train_autoencoder(X_train)
    
    cluster_images(np.concatenate([X_val, X_unlabel], axis=0), encoder)
    
    