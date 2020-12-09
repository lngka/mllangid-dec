from tensorflow.keras import Model
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Flatten
from DECLayer import DECLayer
import tensorflow as tf
import numpy as np
import os


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
        flattened = Flatten()(self.encoder.output)
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
        print(np.array(kmeans.cluster_centers_).shape)

        self.model.get_layer(name='clustering').set_weights(
            [kmeans.cluster_centers_])
        print(f'Set {self.n_lang} cluster centroids initial weights of DECLayer')

    def fit(self, x, y, max_iteration=128, batch_size=64, update_interval=32):
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

            loss = self.model.train_on_batch(x=train_batch, y=train_labels)
            print('==========================================')
            print('train_batch', train_batch.shape)
            print('train_labels.shape', train_labels.shape)
            print('train_labels', train_labels)
            print('ground truth', y[idx])
            print('predicted', self.predict(train_batch))

            # evaluate the clustering performance
            if ite % update_interval == 0:
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

        #self.model.save(f'{self.save_path}/dec', save_format='tf')
        #self.encoder.save(f'{self.save_path}/dec_encoder', save_format='tf')

        # tf.compat.v1.keras.experimental.export_saved_model(
        #     self.model, f'{self.save_path}/dec')
        # tf.compat.v1.keras.experimental.export_saved_model(
        #     self.encoder, f'{self.save_path}/dec_encoder')


class Metrics:
    @staticmethod
    def acc(y_true, y_pred):
        """Calculate clustering accuracy.
        Arguments
            y_true: labels with shape (n_samples,)
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

        acc = sum(w.diagonal()) / y_pred.size
        return acc
