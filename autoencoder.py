
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Dense
from dataset import get_data_set


class AutoEncoder:
    def __init__(self, save_path='./models'):
        autoencoder, encoder = AutoEncoder.build_autoencoder()
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.save_path = save_path

    def get_encoder(self):
        return self.encoder

    @staticmethod
    def build_autoencoder(n_frames=400, fft_bins=100):
        ''' 
        Arguments:
                n_frames: used to shape Input layer
                fft_bins: used to shape Input layer
        Return:
                autoencoder: model of autoencoder
                encoder: model of encoder to initialize features for DECLayer
        '''

        input_layer = Input(shape=(n_frames, fft_bins, 1), dtype=float)
        # encoder
        corrupted_x = Dropout(0.1)(input_layer)
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(corrupted_x)
        h = MaxPooling2D((2, 2), padding='same')(h)

        features = Conv2D(1, (3, 3), padding='same', name='features_layer')(h)

        # decoder
        corrupted_h = Dropout(0.1)(features)
        y = Conv2D(64, (3, 3), activation='relu', padding='same')(corrupted_h)
        y = UpSampling2D((2, 2))(y)
        y = Conv2D(1, (3, 3), padding='same')(y)

        return Model(input_layer, y, name='autoencoder'), Model(input_layer, features, name='encoder')

    @staticmethod
    def load_encoder(path_to_encoder='./models/encoder'):
        return tf.keras.models.load_model(path_to_encoder)

    def fit(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE'):
        self.autoencoder.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=True)

        steps_per_epoch = math.ceil(data.shape[0] / batch_size)
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        self.autoencoder.fit(data, data, batch_size=batch_size,
                             epochs=epochs, steps_per_epoch=steps_per_epoch)

        if save_trained_model:
            self.autoencoder.save(f'{self.save_path}/ae', save_format='tf')
            self.encoder.save(f'{self.save_path}/encoder', save_format='tf')

        # print(autoencoder.get_layer('conv2d').get_weights())
        # print(encoder.get_layer('conv2d').get_weights())
