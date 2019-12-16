from tensorflow import keras as k

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalAveragePooling2D, Dense, concatenate, BatchNormalization
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model


class MeanTeacherModelBN(k.models.Model):
    
    def __init__(self, input_shape, output_shape):
        input0, output0 = self.create_model(input_shape, output_shape)
        super(MeanTeacherModelBN, self).__init__(input0, output0)
        
        self.init_weights = self.get_weights()
    
    def create_model(self, input_shape, output_shape):
        input0 = Input(input_shape)

        kernel_init = 'he_normal'

        net = GaussianNoise(stddev=0.15)(input0)

        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Dropout(rate=0.5)(net)

        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Dropout(rate=0.5)(net)

        net = Conv2D(512, (3, 3), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(256, (1, 1), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = Conv2D(128, (1, 1), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = GlobalAveragePooling2D()(net)
        net = Dense(units=output_shape, activation=None, kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        output0 = Activation('softmax')(net)

        return input0, output0
    
    def reset(self):
        self.set_weights(self.init_weights)
    
    @staticmethod
    def copy(old_model):
        input_shape, output_shape = old_model.input_shape[1:], old_model.output_shape[1]
        new_model = MeanTeacherModelBN(input_shape, output_shape)
        
        new_model.set_weights(old_model.get_weights())
        return new_model
