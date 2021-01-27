
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
        input_layer = Input(shape=(n_frames, fft_bins), dtype=float)
        #x = Flatten()(input_layer)

        # encoder
        h = Dense(500, 'relu')(input_layer)
        h = Dense(500, 'relu')(h)
        h = Dense(2000, 'relu')(h)

        h = Dense(100, 'relu')(h)

        features = Dense(50, name='embeddings')(h)

        # decoder
        h = Dense(100, 'relu')(h)
        h = Dense(2000, 'relu')(h)
        h = Dense(500, 'relu')(h)
        h = Dense(500, 'relu')(h)

        h = Dense(fft_bins)(h)

        y = Reshape(target_shape=(n_frames, fft_bins))(h)

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
            self.autoencoder.summary()

        loss = self.autoencoder.fit(data, epochs=epochs, **kwargs)

        if save_trained_model:
            tf.compat.v1.keras.experimental.export_saved_model(
                self.encoder, f'{self.save_path}/encoder_{self.model_id}')
            # self.encoder.save(f'{self.save_path}/encoder_{self.model_id}')

        return loss
