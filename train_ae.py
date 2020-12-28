import numpy as np
from AutoEncoder import AutoEncoder
from dataset import get_shuffled_data_set
import tensorflow as tf
from tensorflow import keras
import os


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __init__(self, model=None):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.model = model

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

    # def on_train_batch_end(self, batch, logs=None):
    #     logpath, open_mode = LossAndErrorPrintingCallback.get_log_path_and_open_mode()
    #     with open(logpath, open_mode) as text_file:
    #         print("For batch {}, loss is {:7.2f}. \n".format(
    #             batch, logs["loss"]), file=text_file)

    def on_epoch_end(self, epoch, logs=None):
        logpath, open_mode = LossAndErrorPrintingCallback.get_log_path_and_open_mode()
        with open(logpath, open_mode) as text_file:
            print(
                "The average loss for epoch {} is {:7.2f}".format(
                    epoch, logs["loss"]
                ),
                file=text_file
            )

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

''' Step1: Get nice data
'''
data, labels = get_shuffled_data_set()
data = np.expand_dims(data, -1)

''' Step2: Train
'''
autoencoder = AutoEncoder()

log_callback = LossAndErrorPrintingCallback(model=autoencoder.get_encoder())
my_callbacks = [log_callback]

# autoencoder.fit(data, save_trained_model=True, batch_size=1,
#                  epochs=4096, callbacks=my_callbacks)

for i in range(5):
    start = i * 100
    end = start + 100
    d = data[start:end]

    if i == 4:
        loss = autoencoder.fit(
            d, save_trained_model=True, batch_size=100, epochs=2048, callbacks=my_callbacks)
        print('final loss: ', loss)
    else:
        loss = autoencoder.fit(d, save_trained_model=False,
                               batch_size=64, epochs=2048, callbacks=my_callbacks)
    print('trained d:', d.shape)
    print('trained start:', start)
    print('trained end:', end)
