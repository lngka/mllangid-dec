import numpy as np
from LanguageDEC import LanguageDEC
from AutoEncoder import AutoEncoder
from dataset import get_data_set

''' Step0: Get nice data
'''
data, labels = get_data_set()
data = np.expand_dims(data, -1)

''' Step1: Initialization
    1.1: train autoencoder and use the encoder to extract features from data
    1.2: initialize centroids using k_means
'''
# train encoder
# autoencoder = AutoEncoder()
#autoencoder.fit(data, save_trained_model=True, batch_size=10, epochs=1)
#encoder = autoencoder.get_encoder()

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
languageDEC.fit(x=data, y=labels, max_iteration=4, update_interval=1)
