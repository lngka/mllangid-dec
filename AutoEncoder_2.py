
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
        x = Flatten(input_layer)

        # encoder
        corrupted_x = Dropout(0.1)(x)
        h = Dense(10000, 'relu')(x)

        features = Dense(2500)(h)

        # decoder
        corrupted_h = Dropout(0.1)(features)
        h = Dense(10000, 'relu')(corrupted_h)

        h = Dense(n_frames * fft_bins)(h)

        h = Reshape(target_shape=(n_frames, fft_bins, 1))(h)

        return Model(input_layer, y, name='autoencoder'), Model(input_layer, features, name='encoder')

    @staticmethod
    def load_encoder(path_to_encoder=None):
        if path_to_encoder == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path_to_encoder = f'{dir_path}/models/encoder'
        return tf.compat.v1.keras.experimental.load_from_saved_model(path_to_encoder)

    def fit(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE'):
        self.autoencoder.summary()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001)

        steps_per_epoch = math.ceil(data.shape[0] / batch_size)
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        loss = self.autoencoder.fit(data, data, batch_size=batch_size,
                                    epochs=epochs, steps_per_epoch=steps_per_epoch)

        if save_trained_model:
            tf.compat.v1.keras.experimental.export_saved_model(
                self.autoencoder, f'{self.save_path}/ae')
            tf.compat.v1.keras.experimental.export_saved_model(
                self.encoder, f'{self.save_path}/encoder')

        return loss

        # print(autoencoder.get_layer('conv2d').get_weights())
        # print(encoder.get_layer('conv2d').get_weights())
