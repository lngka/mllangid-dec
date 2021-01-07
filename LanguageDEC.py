from tensorflow.keras import Model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.layers import Flatten
from DECLayer import DECLayer
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K


class LanguageDEC:
    '''LanguageDEC Model consist of an encoder to extract features
    and a Deep Embedded Clustering layer (DECLayer) which accepts and 
    assign the features into clusters according to a target distribution
    '''

    def __init__(self, encoder=None, save_path=None, languages=['en', 'de', 'cn', 'fr', 'ru']):
        # creating properties
        if save_path == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_path = f'{dir_path}/models'

        self.save_path = save_path
        self.languages = languages
        self.n_lang = len(languages)

        self.encoder = encoder

        # creating model
        flattened = Flatten(name='flattened_encoder_output')(
            self.encoder.output)
        dec = DECLayer(self.n_lang, name='clustering')
        prediction = dec(flattened)

        self.model = Model(inputs=self.encoder.input, outputs=prediction)

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.summary()

    def extract_features(self, x):
        features = self.encoder.predict(x)
        features = features.reshape((features.shape[0], -1))
        return features

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def calulate_target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, training_data):
        kmeans = KMeans(n_clusters=self.n_lang)
        features = self.extract_features(training_data)

        kmeans.fit_predict(features)
        print(f'Initialized {self.n_lang} cluster centroids')
        print(kmeans.cluster_centers_)

        self.model.get_layer(name='clustering').set_weights(
            [kmeans.cluster_centers_])
        print(f'Set {self.n_lang} cluster centroids initial weights of DECLayer')

    def write_training_log(self, key=None, value=''):
        '''
        Write to log file 
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_path = f'{dir_path}/logs/dec_logs.txt'
        if os.path.exists(log_path):
            open_mode = 'a'  # append if already exists
        else:
            open_mode = 'w'

        with open(log_path, open_mode) as text_file:
            if key == None:
                # write model summary
                self.model.summary(
                    print_fn=lambda x: text_file.write(x + '\n'))

            else:
                print(f'{key}: {value}', file=text_file)

                # write distance between cluster centers
                centroids = self.model.get_layer(
                    name='clustering').get_weights()[0]
                dists = euclidean_distances(centroids)
                print(f'Distances:\n{dists}', file=text_file)

    def fit(self, x, y, max_iteration=128, batch_size=64, update_interval=32, **kwargs):
        self.write_training_log()

        index_array = np.arange(x.shape[0])
        index = 0
        for ite in range(max_iteration):
            q = self.model.predict(x)
            p = self.calulate_target_distribution(q)

            # idx is a list of indices,
            # used to select batch from dataset & labels
            from_index = index * batch_size
            to_index = min((index+1) * batch_size, x.shape[0])
            idx = index_array[from_index:to_index]

            train_x = x[idx]
            train_y = p[idx]

            loss = self.model.train_on_batch(
                x=train_x, y=train_y, **kwargs)

            # evaluate the clustering performance
            if ite % update_interval == 0:
                self.write_training_log('=======================ite', ite)
                self.write_training_log('loss: ', loss)

                q = self.model.predict(x)
                y_pred = q.argmax(1)

                # not really accuracy, just trigger logging for now
                Metrics.acc(y, y_pred, languages=self.languages)

            # update index
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        tf.keras.models.save_model(
            self.model, f'{self.save_path}/dec', save_format='h5')


class Metrics:
    @staticmethod
    def write_training_log(key='', value=''):
        '''
        Write to log file 
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_path = f'{dir_path}/logs/dec_logs.txt'
        if os.path.exists(log_path):
            open_mode = 'a'  # append if already exists
        else:
            open_mode = 'w'

        with open(log_path, open_mode) as text_file:
            print(f'{key}{value}', file=text_file)

    @staticmethod
    def acc(y_true, y_pred, languages=['en', 'de', 'cn', 'fr', 'ru']):
        """Calculate clustering accuracy.
        Arguments
            y_true: labels with shape (n_samples,)
                    en-0, de-1, cn-2, fr-3, ru-4
            y_pred: predicted labels with shape (n_samples,)
        Return
            accuracy
        """
        y_true = y_true.astype(np.int64)

        print('y_true:', y_true.shape)
        print('y_pred:', y_pred.shape)

        assert y_pred.size == y_true.size

        #D = max(y_pred.max(), y_true.max()) + 1
        n_lang = len(languages)
        w = np.zeros((n_lang, n_lang), dtype=np.int64)

        for i in range(y_pred.size):
            # w[i, j] = count of prediction&groundtruth pair i&j
            w[y_pred[i], y_true[i]] += 1

        Metrics.write_training_log("Prediction results", " ")
        Metrics.write_training_log("(pred_axis, truth_axis) \n", w)

        true_labelmap = tf.argmax(w, axis=0)
        likelihood = tf.reduce_max(w, axis=0) / tf.reduce_sum(w, axis=0)

        for i in range(len(languages)):
            Metrics.write_training_log(
                f'True label {languages[i]} classified as ', f'{true_labelmap[i]}, likelihood {likelihood[i]}')

        acc = sum(w.diagonal()) / y_pred.size
        return acc