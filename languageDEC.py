from tensorflow.keras import Model
from sklearn.cluster import KMeans
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

    def __init__(self, encoder=None, n_lang=5, save_path=None):
        # creating properties
        if save_path == None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            save_path = f'{dir_path}/models'

        self.save_path = save_path
        self.n_lang = n_lang
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
                self.model.summary(
                    print_fn=lambda x: text_file.write(x + '\n'))
            else:
                print(f'{key}: {value}', file=text_file)

    def fit(self, x, y, max_iteration=128, batch_size=64, update_interval=32, **kwargs):
        self.write_training_log()

        index_array = np.arange(x.shape[0])
        index = 0
        loss = 0
        for ite in range(max_iteration):
            q = self.model.predict(x)
            p = self.calulate_target_distribution(q)

            # idx is a list of indices,
            # used to select batch from dataset & labels
            from_index = index * batch_size
            to_index = min((index+1) * batch_size, x.shape[0])
            idx = index_array[from_index:to_index]

            train_batch = x[idx]
            train_labels = p[idx]

            loss = self.model.train_on_batch(
                x=train_batch, y=train_labels, **kwargs)

            # evaluate the clustering performance
            if ite % update_interval == 0:
                print('=================BATCH====================')
                print('train_batch.shape', train_batch.shape)
                print('train_labels.shape', train_labels.shape)
                #print('train_labels', train_labels)
                #print('ground truth', y[idx])
                #print('predicted', self.predict(train_batch))
                self.write_training_log('ite', ite)
                self.write_training_log('train_batch.shape', train_batch.shape)
                self.write_training_log(
                    'train_labels.shape', train_labels.shape)

                y_pred = q.argmax(1)
                acc = np.round(Metrics.acc(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f' % (ite, acc),
                      '; loss = ', loss)
            # update index
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        print("Done fitting")
        tf.keras.models.save_model(
            self.model, f'{self.save_path}/dec', save_format='h5')
        tf.keras.models.save_model(
            self.encoder, f'{self.save_path}/dec_encoder', save_format='h5')


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
            print(f'{key}: {value}', file=text_file)

    @staticmethod
    def acc(y_true, y_pred):
        """Calculate clustering accuracy.
        Arguments
            y_true: labels with shape (n_samples,)
                    en-0, de-1, cn-2, fr-3, ru-4
            y_pred: predicted labels with shape (n_samples,)
        Return
            accuracy
        """
        y_true = y_true.astype(np.int64)

        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(y_pred.size):
            # w[i, j] = count of prediction&groundtruth pair i&j
            w[y_pred[i], y_true[i]] += 1

        Metrics.write_training_log("Prediction results", " ")
        Metrics.write_training_log("w", w)

        labelmap = tf.argmax(w, axis=0)
        class_count = tf.reduce_max(w, axis=0)

        Metrics.write_training_log("labelmap", labelmap)
        Metrics.write_training_log("class_count", class_count)

        for i in range(5):
            #print(f'True label {i} classified as {labelmap[i]}')
            #print(f'Count: {class_count[i]}')
            Metrics.write_training_log(
                f'True label {i} classified as', f'{labelmap[i]}')
            Metrics.write_training_log("Count", class_count[i])

        acc = sum(w.diagonal()) / y_pred.size
        return acc
