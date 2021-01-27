from tensorflow import keras
import os
import tensorflow as tf


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def __init__(self, model=None, languages=['en', 'de', 'cn', 'fr', 'ru'],  model_id=''):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.model = model
        self.languages = languages
        self.model_id = model_id

    def get_log_path_and_open_mode(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_path = f'{dir_path}/logs/ae_logs_{self.model_id}.txt'

        if os.path.exists(log_path):
            open_mode = 'a'  # append if already exists
        else:
            open_mode = 'w'  # make a new file if not
        return log_path, open_mode

    def on_train_begin(self, logs=None):
        logpath, open_mode = self.get_log_path_and_open_mode()
        with open(logpath, open_mode) as text_file:
            self.model.summary(print_fn=lambda x: text_file.write(x + '\n'))
            print(self.languages,  file=text_file)

    # def on_train_end(self, logs=None):
    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     save_path = f'{dir_path}/models'
    #     tf.compat.v1.keras.experimental.export_saved_model(
    #         self.model, f'{save_path}/encoder_{self.model_id}')

    def on_epoch_end(self, epoch, logs=None):
        logpath, open_mode = self.get_log_path_and_open_mode()
        with open(logpath, open_mode) as text_file:
            print(
                "The average loss for epoch {} is {:7.3f}".format(
                    epoch, logs["loss"]
                ),
                file=text_file
            )


def ModelCheckpoint(checkpoint_filepath='', save_weights_only=True, monitor='loss', mode='min', **kwargs):
    if not os.path.exists(os.path.dirname(checkpoint_filepath)):
        os.makedirs(os.path.dirname(checkpoint_filepath))

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=save_weights_only,
        monitor=monitor,
        mode=mode,
        save_best_only=True, **kwargs)
