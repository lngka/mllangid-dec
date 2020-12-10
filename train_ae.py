import numpy as np
from AutoEncoder_2 import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf


''' Step1: Get nice data
'''
data, labels = get_shuffled_data_set()
data = np.expand_dims(data, -1)

''' Step2: Train
'''
autoencoder = AutoEncoder()
for i in range(5):
    start = i * 100
    end = start + 100
    d = data[start:end]

    if i == 4:
        loss = autoencoder.fit(
            d, save_trained_model=True, batch_size=64, epochs=512)
        print('final loss: ', loss)
    else:
        loss = autoencoder.fit(d, save_trained_model=False,
                               batch_size=64, epochs=512)
    print('trained d:', d.shape)
    print('trained start:', start)
    print('trained end:', end)
