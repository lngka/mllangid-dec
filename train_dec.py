from LanguageDEC import LanguageDEC
import tensorflow as tf
import numpy as np
from AutoEncoder_2 import AutoEncoder
from dataset import get_shuffled_data_set
import os

tf.compat.v1.enable_eager_execution()

MODEL_ID = '80_3L'  # use to name log txt file and save model

''' Step0: Get nice data
'''
#languages = ['en', 'de', 'cn', 'fr', 'ru']
languages = ['en', 'de', 'cn']
data, classes, data_test, classes_test = get_shuffled_data_set(
    languages,  split=True)


# add dim when train with conv encoder
#data = np.expand_dims(data, -1)
#data_test = np.expand_dims(data_test, -1)


''' Step1: Initialization
    1.1: load pre trained encoder to extract features from data
    1.2: initialize centroids using k_means
'''
autoencoder = AutoEncoder(n_frames=400, fft_bins=40)
encoder = autoencoder.load_encoder(model_id=MODEL_ID)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# checkpoint_filepath = f'{dir_path}/model_checkpoints/ae_70/weights.1925.hdf5'
# autoencoder.autoencoder.load_weights(checkpoint_filepath)
# encoder = autoencoder.get_encoder()


# initialize centroid using k_means
languageDEC = LanguageDEC(
    encoder=encoder, languages=languages, model_id=MODEL_ID, robust=True)
languageDEC.initialize(data, classes)


''' Step2: Optimze model to target distribution
    max_iteration: how many times to go through whole data set, aka. epochs
    update_interval: 
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
languageDEC.compile(optimizer=optimizer, loss='kld')
languageDEC.fit(x=data, y=classes, max_iteration=4096,
                update_interval=16, x_test=data_test, y_test=classes_test)
