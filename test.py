from dataset import get_data_set
from LanguageDEC import LanguageDEC
import os
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = f'{dir_path}/models'

dataset, classes = get_data_set(['en'])

encoder = tf.keras.models.load_model(f'{save_path}/dec_encoder')
languageDEC = tf.keras.models.load_model(f'{save_path}/dec')

languageDEC.summary()
