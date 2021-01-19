import numpy as np
from LanguageDEC import LanguageDEC
from AutoEncoder import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

MODEL_ID = '61'  # use to name log txt file and save model

''' Step0: Get nice data
'''
languages = ['en', 'de', 'cn', 'fr', 'ru']
#languages = ['en','cn', 'fr', 'ru']

data, labels = get_shuffled_data_set(languages)
data = np.expand_dims(data, -1)


''' Step1: Initialization
    1.1: load pre trained encoder to extract features from data
    1.2: initialize centroids using k_means
'''


# load pre-trained encoder
autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder(model_id=MODEL_ID)

# initialize centroid using k_means
languageDEC = LanguageDEC(
    encoder=encoder, languages=languages, model_id=MODEL_ID)
languageDEC.initialize(data)


''' Step2: Optimze model to target distribution
    max_iteration: how many times to go through whole data set, aka. epochs
    update_interval: 
'''
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
languageDEC.compile(optimizer=optimizer, loss='kld')
languageDEC.fit(x=data, y=labels, max_iteration=1024,
                update_interval=16, batch_size=100)
