import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout

FFT_LENGTH = 198  # to have 100 fft bins
FFT_BINS = FFT_LENGTH // 2 + 1

SAMPLING_RATE = 8000
WIN_SAMPLES = int(SAMPLING_RATE * 0.025)
HOP_SAMPLES = int(SAMPLING_RATE * 0.010)
N_FRAMES = 400


def autoencoder():
    input_layer = Input(shape=(N_FRAMES, FFT_BINS, 1), dtype=float)
    # encoder
    corrupted_x = Dropout(0.1)(input_layer)
    h = Conv2D(64, (3, 3), activation='relu', padding='same')(corrupted_x)
    h = MaxPooling2D((2, 2), padding='same', name='feature_layer')(h)

    # decoder
    corrupted_h = Dropout(0.1)(h)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(corrupted_h)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(1, (3, 3), padding='same')(y)

    return Model(input_layer, y, name='autoencoder'), Model(input_layer, h, name='encoder')


if __name__ == "__main__":
    autoencoder, encoder = autoencoder()
    autoencoder.summary()
    cn_stfts = np.load('./8K/cn_stfts.npy')

    cn_stfts = tf.expand_dims(cn_stfts, axis=-1)

    x_train = y_train = cn_stfts[:80, ]
    x_val = y_val = cn_stfts[-20:, ]

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=True)

    autoencoder.compile(loss='MSE', optimizer=optimizer)
    autoencoder.fit(x_train, y_train, batch_size=8,
                    epochs=1, validation_data=(x_val, y_val))
    # autoencoder.save('./models/ae')
    # encoder.save('./models/encoder')
