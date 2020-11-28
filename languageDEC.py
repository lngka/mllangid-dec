from tensorflow.keras import Model
import numpy as np
from autoencoder import autoencoder
from sklearn.cluster import KMeans
from DECLayer import DECLayer


class LanguageDEC:
    def __init__(self, encoder=None, n_lang=5):
        self.encoder = encoder
        self.n_lang = n_lang
        # prepare DEC model
        prediction = DECLayer(self.n_lang, name='clustering')(
            self.encoder.output)

        self.model = Model(inputs=self.encoder.input, outputs=prediction)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def initialize(self, training_data):
        kmeans = KMeans(n_clusters=self.n_lang)
        features = self.extract_features(training_data)
        kmeans.fit_predict(features)

        self.model.get_layer(name='clustering').set_weights(
            [kmeans.cluster_centers_])

        print(kmeans.cluster_centers_)
        print(np.array(kmeans.cluster_centers_).shape)
