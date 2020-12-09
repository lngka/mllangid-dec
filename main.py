import numpy as np
from LanguageDEC import LanguageDEC
from AutoEncoder import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

''' Step0: Get nice data
'''
data, labels = get_shuffled_data_set()
data = np.expand_dims(data, -1)

''' Step1: Initialization
    1.1: train autoencoder and use the encoder to extract features from data
    1.2: initialize centroids using k_means
'''
# train encoder
# autoencoder = AutoEncoder()
# for i in range(5):
#     start = i * 100
#     end = start + 100
#     d = data[start:end]

#     if i == 4:
#         loss = autoencoder.fit(
#             d, save_trained_model=True, batch_size=64, epochs=512)
#         print('final loss: ', loss)
#     else:
#         loss = autoencoder.fit(d, save_trained_model=False,
#                                batch_size=64, epochs=512)
#     print('trained d:', d.shape)
#     print('trained start:', start)
#     print('trained end:', end)


# encoder = autoencoder.get_encoder()


# load pre-trained encoder
autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder()

# initialize centroid using k_means
languageDEC = LanguageDEC(encoder=encoder, n_lang=5)
languageDEC.initialize(data)


''' Step2: Optimze model to target distribution
    max_iteration: how many times to go through whole data set, aka. epochs
    update_interval: 
'''
languageDEC.compile(optimizer='sgd', loss='kld')
languageDEC.fit(x=data, y=labels, max_iteration=1, update_interval=1)
