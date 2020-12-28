
import tensorflow as tf
import numpy as np
import math
import os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Dense
from dataset import get_data_set


class AutoEncoder:
    def __init__(self, save_path=None):
        if save_path == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_path = f'{dir_path}/models'

        self.save_path = save_path

        autoencoder, encoder = AutoEncoder.build_autoencoder()
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.already_compiled = False

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
        h = Conv2D(32, (3, 3), activation='relu', padding='same')(corrupted_x)
        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)

        features = Conv2D(1, (3, 3), padding='same', name='features_layer')(h)

        # decoder
        corrupted_h = Dropout(0.1)(features)

        h = Conv2D(128, (3, 3), activation='relu', padding='same')(corrupted_h)

        h = UpSampling2D((2, 2))(h)
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)

        h = UpSampling2D((2, 2))(h)
        h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)

        y = Conv2D(1, (3, 3), padding='same')(h)

        return Model(input_layer, y, name='autoencoder'), Model(input_layer, features, name='encoder')

    @staticmethod
    def load_encoder(path_to_encoder=None):
        if path_to_encoder == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path_to_encoder = f'{dir_path}/models/encoder_8'
        return tf.compat.v1.keras.experimental.load_from_saved_model(path_to_encoder)

    def fit(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE', **kwargs):
        self.autoencoder.summary()
        steps_per_epoch = math.ceil(data.shape[0] / batch_size)

        if self.already_compiled:
            loss = self.autoencoder.fit(data, data, batch_size=batch_size,
                                        epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
        else:
            # compile and fit if not already compiled
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.autoencoder.compile(loss=loss, optimizer=optimizer)
            self.already_compiled = True
            loss = self.autoencoder.fit(data, data, batch_size=batch_size,
                                        epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)

        if save_trained_model:
            tf.compat.v1.keras.experimental.export_saved_model(
                self.autoencoder, f'{self.save_path}/ae')
            tf.compat.v1.keras.experimental.export_saved_model(
                self.encoder, f'{self.save_path}/encoder')

        return loss
