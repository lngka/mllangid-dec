
import tensorflow as tf
import numpy as np
import math
import os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Dense
from dataset import get_data_set


class AutoEncoder:
    def __init__(self, save_path=None, n_frames=400, fft_bins=100, model_id=''):
        if save_path == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_path = f'{dir_path}/models'

        self.save_path = save_path
        self.model_id = model_id
        autoencoder, encoder = AutoEncoder.build_autoencoder(
            n_frames=n_frames, fft_bins=fft_bins)

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.already_compiled = False

    def get_encoder(self):
        return self.encoder

    def get_autoencoder(self):
        return self.autoencoder

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
        corrupted_x = Dropout(0)(input_layer)
        h = Conv2D(32, (3, 3), activation='relu', padding='same')(corrupted_x)
        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)

        h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
        h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)

        last_pooling_shape = h.shape

        h = Flatten()(h)
        flat_shape = h.shape
        h = Dense(2000)(h)
        features = Dense(100, name='embeddings')(h)
        h = Dense(2000)(features)
        h = Dense(flat_shape[1])(h)

        h = Reshape(target_shape=last_pooling_shape[1:])(h)

        h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)

        # decoder
        corrupted_h = Dropout(0)(h)

        h = Conv2D(128, (3, 3), activation='relu', padding='same')(corrupted_h)
        h = UpSampling2D((2, 2))(h)

        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)

        h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)

        y = Conv2D(1, (3, 3), padding='same')(h)

        return Model(input_layer, y, name='autoencoder'), Model(input_layer, features, name='encoder')

    @staticmethod
    def load_encoder(path_to_encoder=None, model_id=''):
        if path_to_encoder == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path_to_encoder = f'{dir_path}/models/encoder_{model_id}'
        return tf.compat.v1.keras.experimental.load_from_saved_model(path_to_encoder)

    def fit(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE', **kwargs):

        if self.already_compiled == False:
            # compile if not already compiled
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.autoencoder.compile(loss=loss, optimizer=optimizer)
            self.already_compiled = True

        loss = self.autoencoder.fit(data, epochs=epochs, **kwargs)

        if save_trained_model:
            tf.compat.v1.keras.experimental.export_saved_model(
                self.encoder, f'{self.save_path}/encoder_{self.model_id}')
            # self.encoder.save(f'{self.save_path}/encoder_{self.model_id}')

        return loss
