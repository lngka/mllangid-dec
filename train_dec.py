import numpy as np
from LanguageDEC import LanguageDEC
from AutoEncoder import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
''' Step0: Get nice data
'''
languages = ['en', 'de', 'cn', 'fr', 'ru']
#languages = ['en','cn', 'fr', 'ru']

data, labels = get_shuffled_data_set(languages)
data = np.expand_dims(data, -1)

# data = np.full(shape=(10, 400, 100, 1), fill_value=0.645)  # fake data to test
#data = np.random.rand(10, 400, 100, 1)
#np.save('./testdata.npy', data)
#labels = np.full(shape=(10,), fill_value=3)
#data = np.load('./testdata.npy')

''' Step1: Initialization
    1.1: load pre trained encoder to extract features from data
    1.2: initialize centroids using k_means
'''


# load pre-trained encoder
autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder()

# initialize centroid using k_means
languageDEC = LanguageDEC(encoder=encoder, languages=languages)
languageDEC.initialize(data)


''' Step2: Optimze model to target distribution
    max_iteration: how many times to go through whole data set, aka. epochs
    update_interval: 
'''
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
languageDEC.compile(optimizer=optimizer, loss='kld')
languageDEC.fit(x=data, y=labels, max_iteration=512,
                update_interval=16, batch_size=100)
