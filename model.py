import keras as k
from keras.layers import Input, Dense, Activation, Dropout
import keras.regularizers as regularizers

import numpy as np


class MNISTModel(k.models.Model):
    
    def __init__(self, input_dim, output_dim):
        input0, output0 = self._create_model(input_dim, output_dim)
        super(MNISTModel, self).__init__(input0, output0)
        
        
    def _create_model(self, input_dim, output_dim):
        input_dim = np.prod(input_dim)
        
        input0 = Input((input_dim, ))
        
        x = Dense(256)(input0)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(output_dim)(x)
        output0 = Activation('softmax')(x)
        
        return input0, output0
    
    def reset(self):
        for layer in self.layers:
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                # find the corresponding variable
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
    
