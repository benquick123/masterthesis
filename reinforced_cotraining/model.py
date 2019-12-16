import keras as k
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Embedding, LSTM


class SimpleModel(k.models.Model):
    
    def __init__(self, input_dim, output_dim, layer_dims):
        input0, output0 = self._create_model(input_dim, output_dim, layer_dims)
        super(SimpleModel, self).__init__(input0, output0)
    
    def _create_model(self, input_dim, output_dim, layer_dims):
        input0 = Input(input_dim)
        
        x = Dense(layer_dims[0], activation="relu")(input0)
        x = Dense(layer_dims[1], activation="relu")(x)
        
        output0 = Dense(output_dim, activation="softmax")(x)
        
        return input0, output0
    
    def reset(self):
        for layer in self.layers:
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                # find the corresponding variable
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
    

class LSTMTextModel(k.models.Model):
    
    def __init__(self, input_dim, output_dim):
        input0, output0 = self._create_model(input_dim, output_dim)
        super(LSTMTextModel, self).__init__(input0, output0)
        
    def _create_model(self, input_dim, output_dim):
        input0 = Input(shape=input_dim, dtype='int32')
        
        embedding_layer = Embedding(20000, 50, input_length=input_dim[0], trainable=True)

        embedded_sequences = embedding_layer(input0)

        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        output0 = Dense(output_dim, activation='softmax')(x)
        
        return input0, output0
    
    def reset(self):
        for layer in self.layers:
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                # find the corresponding variable
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))

    # for x2: optimizer = Adam(lr=0.001)
    # for x1: optimizer = Adam(lr=0.0005)

if __name__ == "__main__":
    
    model = SimpleModel((50, ), 2)
    model.summary()
