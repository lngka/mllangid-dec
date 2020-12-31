
import tensorflow as tf
import numpy as np
import math
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Dense
from dataset import get_data_set


class AutoEncoder:
    def __init__(self, save_path=None, n_frames=400, fft_bins=100):
        if save_path == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_path = f'{dir_path}/models'

        self.save_path = save_path

        autoencoder, encoder = AutoEncoder.build_autoencoder(
            n_frames=n_frames, fft_bins=fft_bins)
        autoencoder.summary()

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
        resnet = ResNet50(input_shape=input_layer.shape[1:],
                          weights=None, include_top=True)

        features = resnet(input_layer)

        h = Dense(2048)(features)
        h = Reshape(target_shape=(1, 1, 2048))(h)
        h = UpSampling2D(size=(50, 5))(h)
        h = Conv2D(512, (3, 3), padding='same', activation='relu')(h)

        h = UpSampling2D(size=(2, 2))(h)
        h = Conv2D(256, (3, 3), padding='same', activation='relu')(h)

        h = UpSampling2D(size=(2, 2))(h)
        h = Conv2D(128, (3, 3), padding='same', activation='relu')(h)

        h = UpSampling2D(size=(2, 2))(h)
        h = Conv2D(64, (3, 3), padding='same', activation='relu')(h)

        y = Conv2D(1, (2, 2), padding='same')(h)

        return Model(input_layer, y, name='autoencoder'), Model(input_layer, features, name='encoder')

    @staticmethod
    def load_encoder(path_to_encoder=None):
        if path_to_encoder == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path_to_encoder = f'{dir_path}/models/encoder'
        return tf.compat.v1.keras.experimental.load_from_saved_model(path_to_encoder)

    def fit(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE', **kwargs):
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

    def fit_batch(self, data, save_trained_model=False, batch_size=10, epochs=1, loss='MSE', **kwargs):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        self.already_compiled = True

        for i in range(5):
            start = i * 100
            end = start + 100
            d = data[start:end]

            if i == 4:
                loss = self.autoencoder.fit(
                    d, d, batch_size=batch_size, epochs=epochs, **kwargs)

                if save_trained_model:
                    tf.compat.v1.keras.experimental.export_saved_model(
                        self.autoencoder, f'{self.save_path}/ae')
                    tf.compat.v1.keras.experimental.export_saved_model(
                        self.encoder, f'{self.save_path}/encoder')
            else:
                loss = self.autoencoder.fit(
                    d, d, batch_size=batch_size, epochs=epochs, **kwargs)

        return loss
