from dataset import get_data_set
from LanguageDEC import LanguageDEC
from DECLayer import DECLayer
import os
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = f'{dir_path}/models'
encoder = tf.keras.models.load_model(f'{save_path}/dec_encoder')
languageDEC = tf.keras.models.load_model(
    f'{save_path}/dec', custom_objects={'DECLayer': DECLayer})

languageDEC.summary()

dataset, classes = get_data_set(['de'])

dataset = tf.expand_dims(dataset, -1)

predictions = languageDEC.predict(dataset, steps=1)
print(predictions.argmax(1))
