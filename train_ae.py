
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

from AutoEncoder import AutoEncoder
from dataset import get_data_set
from Callbacks import LossAndErrorPrintingCallback, ModelCheckpoint

MODEL_ID = '61'  # use to name log txt file and save model

''' Step1: Get nice data
'''
#languages = ['en', 'de', 'cn', 'fr', 'ru']
languages = ['en', 'de', 'cn']
dataset_train, classes_train, dataset_test, classes_test = get_data_set(
    languages,  split=False)

data = np.expand_dims(dataset_train, -1)
data_test = np.expand_dims(dataset_test, -1)

dataset_train = tf.data.Dataset.from_tensor_slices(
    (data, data))
dataset_train = dataset_train.shuffle(100).batch(100)

''' Step2: Define callback
'''
# to save checkpoints
dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_filepath = f'{dir_path}/model_checkpoints/ae/weights' + \
    '.{epoch:02d}.hdf5'
model_checkpoint_callback = ModelCheckpoint(checkpoint_filepath)

# to log
autoencoder = AutoEncoder(n_frames=400, fft_bins=40, model_id=MODEL_ID)
log_callback = LossAndErrorPrintingCallback(
    model=autoencoder.get_encoder(), languages=languages, model_id=MODEL_ID)

# together
my_callbacks = [log_callback, model_checkpoint_callback]

''' Step3: Train
'''
autoencoder.fit(dataset_train, epochs=512, callbacks=my_callbacks)
