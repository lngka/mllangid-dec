import numpy as np
from AutoEncoder import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf
from tensorflow import keras
import os


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __init__(self, model=None, languages=['en', 'de', 'cn', 'fr', 'ru']):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.model = model
        self.languages = languages

    @staticmethod
    def get_log_path_and_open_mode():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_path = f'{dir_path}/logs/ae_logs.txt'

        if os.path.exists(log_path):
            open_mode = 'a'  # append if already exists
        else:
            open_mode = 'w'  # make a new file if not
        return log_path, open_mode

    def on_train_begin(self, logs=None):
        logpath, open_mode = LossAndErrorPrintingCallback.get_log_path_and_open_mode()
        with open(logpath, open_mode) as text_file:
            self.model.summary(print_fn=lambda x: text_file.write(x + '\n'))
            print(self.languages,  file=text_file)

    def on_epoch_end(self, epoch, logs=None):
        logpath, open_mode = LossAndErrorPrintingCallback.get_log_path_and_open_mode()
        with open(logpath, open_mode) as text_file:
            print(
                "The average loss for epoch {} is {:7.2f}".format(
                    epoch, logs["loss"]
                ),
                file=text_file
            )


''' Step1: Get nice data
'''
languages = ['en', 'de', 'cn', 'fr', 'ru']
data, labels = get_shuffled_data_set(languages)
data = np.expand_dims(data, -1)
#data = np.expand_dims(data, 1)

dataset = tf.data.Dataset.from_tensor_slices(
    (data, data))
dataset = dataset.shuffle(100).batch(100)


''' Step2: Define callback
'''
autoencoder = AutoEncoder(n_frames=400, fft_bins=40)


dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_filepath = f'{dir_path}/model_checkpoints/ae/weights' + \
    '.{epoch:02d}.hdf5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

log_callback = LossAndErrorPrintingCallback(
    model=autoencoder.get_encoder(), languages=languages)

my_callbacks = [log_callback, model_checkpoint_callback]

''' Step3: Train
'''


autoencoder.fit(dataset, save_trained_model=True,
                epochs=512, callbacks=my_callbacks)

# autoencoder.fit_batch(data, save_trained_model=True, batch_size=100,
#                       epochs=512, callbacks=my_callbacks)
