import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras as k

class ResNet(k.models.Model):
    
    def __init__(self, input_shape, output_shape, architecture=(2, 2, 2, 2)):
        input0, output0 = self.create_model(input_shape, output_shape, architecture)
        super(ResNet, self).__init__(inputs=input0, outputs=output0)

        self.init_weights = self.get_weights()


    def create_model(self, input_shape, output_shape, architecture):
        input0 = k.layers.Input(input_shape)
        conv0 = k.layers.Conv2D(64, 7, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(input0)
        bn0 = k.layers.BatchNormalization(axis=3)(conv0)
        relu0 = k.layers.Activation("relu")(bn0)

        pool0 = k.layers.MaxPool2D((3, 3), strides=2, padding="same")(conv0)

        block = pool0
        filters = 64

        for i, repetitions in enumerate(architecture):
            for j in range(repetitions):
                shortcut = block

                block = k.layers.BatchNormalization(axis=3)(block)
                block = k.layers.Activation("relu")(block)

                if i == 0 and j == 0:
                    shortcut = k.layers.Conv2D(filters, 1, strides=1, padding="valid", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)
                    block = k.layers.Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)
                elif j == 0:
                    shortcut = k.layers.Conv2D(filters, 1, strides=2, padding="valid", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)
                    block = k.layers.Conv2D(filters, 3, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)
                else:
                    block = k.layers.Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)

                block = k.layers.BatchNormalization(axis=3)(block)
                block = k.layers.Activation("relu")(block)
                block = k.layers.Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=k.regularizers.l2(0.0001))(block)
                
                block = k.layers.Add()([shortcut, block])
            filters *= 2
        
        bn1 = k.layers.BatchNormalization(axis=3)(block)
        relu1 = k.layers.Activation("relu")(bn1)

        pool1 = k.layers.AveragePooling2D((1, 1), strides=1)(relu1)
        flattened0 = k.layers.Flatten()(pool1)
        dense0 = k.layers.Dense(output_shape, kernel_initializer="he_normal", activation="softmax")(flattened0)

        return input0, dense0

    
    def reset(self):
        self.set_weights(self.init_weights)
        
    @staticmethod
    def copy(old_model):
        input_shape, output_shape = old_model.input_shape[1:], old_model.output_shape[1]
        new_model = SimpleConvNet(input_shape, output_shape)
        
        new_model.set_weights(old_model.get_weights())
        return new_model


class SimpleConvNet(k.models.Model):

    def __init__(self, input_shape, output_shape):
        input0, output0 = self.create_model(input_shape, output_shape)
        super(SimpleConvNet, self).__init__(input0, output0)

        self.init_weights = self.get_weights()

    
    def create_model(self, input_shape, output_shape):
        input0 = k.layers.Input(input_shape)
        x = k.layers.BatchNormalization()(input0)
            
        x = k.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input0)
        x = k.layers.MaxPooling2D(pool_size=(2, 2))(x) 
        
        """
        x = k.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = k.layers.MaxPooling2D(pool_size=(2, 2))(x)    
        x = k.layers.Dropout(0.25)(x)
        """
        
        x = k.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = k.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = k.layers.Dropout(0.25)(x)
                
        x = k.layers.Flatten()(x)
        
        x = k.layers.Dense(256, activation='relu')(x)    
        x = k.layers.Dropout(0.5)(x)

        output0 = k.layers.Dense(output_shape, activation="softmax")(x)
        return input0, output0
    
    def reset(self):
        self.set_weights(self.init_weights)
        
    @staticmethod
    def copy(old_model):
        input_shape, output_shape = old_model.input_shape[1:], old_model.output_shape[1]
        new_model = SimpleConvNet(input_shape, output_shape)
        
        new_model.set_weights(old_model.get_weights())
        return new_model

class MeanTeacherModel(k.models.Model):
    
    def __init__(self, input_shape, output_shape):
        input0, output0 = self.create_model(input_shape, output_shape)
        super(SimpleConvNet, self).__init__(input0, output0)
        
        self.init_weights = self.get_weights()
    
    def create_model(self, input_shape, output_shape):
        input0 = k.layers.Input(input_shape)
        
        # normalization
        x = k.layers.LayerNormalization(center=False, scale=False)(input0)
        
        x = k.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(x)
        x = k.layers.Conv2D(128)(x)
        x = k.layers.Conv2D(128)(x)
        x = k.layers.MaxPooling2D(pool_size=(2, 2))(x)
        
    
    def reset(self):
        self.set_weights(self.init_weights)
    
    @staticmethod
    def copy(old_model):
        input_shape, output_shape = old_model.input_shape[1:], old_model.output_shape[1]
        new_model = SimpleConvNet(input_shape, output_shape)
        
        new_model.set_weights(old_model.get_weights())
        return new_model