import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Reshape, Dense


input_layer = Input(shape=(400, 40, 1), dtype=float)
resnet = ResNet50(input_shape=input_layer.shape[1:],
                  weights=None, include_top=True)
resnet.summary()


x = input_layer
features = resnet(x)

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

y = Conv2D(1, (3, 3), padding='same')(h)

autoencoder = Model(input_layer, y, name='resnet_ae')
encoder = Model(x, features, name='encoder')
autoencoder.summary()
encoder.summary()
