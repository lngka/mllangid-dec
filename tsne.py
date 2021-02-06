import pandas as pd
import numpy as np
import os
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt

from AutoEncoder_2 import AutoEncoder
from dataset import get_shuffled_data_set, get_data_set
from helpers import robust_mahalanobis_method, mahalanobis_method


''' Step0: Config flags
'''
PLOT_CENTROIDS = True
PLOT_CUSTOM_CENTROIDS = True

''' Step1: Get nice data
'''
#languages = ['en', 'de', 'cn', 'fr', 'ru']
languages = ['en', 'cn']
data, labels, _, _ = get_data_set(languages, split=False)
#data = np.expand_dims(data, -1)

''' Step2: Predict embedded data
'''
dir_path = os.path.dirname(os.path.realpath(__file__))
autoencoder = AutoEncoder(n_frames=400, fft_bins=40)

encoder = autoencoder.load_encoder(model_id='70')

# autoencoder.autoencoder.load_weights(
#    f'{dir_path}/model_checkpoints/ae_70/weights.1925.hdf5')

encoder = autoencoder.get_encoder()
encoder.load_weights('model_checkpoints/dec_70/trained_encoder_ite1008.h5')

y = labels
X = encoder.predict(data)
# shape = n_samples x embeddings, use when embedding is 2D
X = X.reshape((X.shape[0], -1))


''' Step3: Get centroids
'''
if PLOT_CUSTOM_CENTROIDS:
    custom_centroids = list()

    for i in range(len(languages)):
        lang = languages[i]
        label = i
        lang_x = X[y == i, ]
        # is_inlier = IsolationForest(
        #     random_state=0, max_features=50, contamination=0.2).fit_predict(lang_x)

        df = pd.DataFrame(lang_x)
        outlier, md = robust_mahalanobis_method(df)
        is_inlier = np.ones(lang_x.shape[0], dtype=int)
        is_inlier[outlier] = 0

        exit()

        lang_x = lang_x[is_inlier == 1]
        lang_centroid = np.average(lang_x, axis=0)
        custom_centroids.append(lang_centroid)
    custom_centroids = np.array(custom_centroids)

    #kmeans = KMeans(n_clusters=len(languages), init=custom_centroids)
    # kmeans.fit_predict(X)
    #custom_centroids = kmeans.cluster_centers_

    l_centroids = len(languages)
    n_centroids = custom_centroids.shape[0]
    y_centroids = np.full((n_centroids, ),  fill_value=l_centroids)

    languages.append('custom_centroids')
    X = np.concatenate((custom_centroids, X), axis=0)
    y = np.concatenate((y_centroids, y), axis=0)

if PLOT_CENTROIDS:
    centroids = np.load('model_checkpoints/dec_70/centroids_ite1008.npy')
    centroids = centroids.reshape((-1, X.shape[-1]))

    l_centroids = len(languages)
    n_centroids = centroids.shape[0]
    y_centroids = np.full((n_centroids, ),  fill_value=l_centroids)

    languages.append('centroids')
    X = np.concatenate((centroids, X), axis=0)
    y = np.concatenate((y_centroids, y), axis=0)


''' Step4: Plot TSNE 
'''
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

indices = range(len(languages))

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'black', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, color, lang in zip(indices, colors, languages):
    z = 0
    m = 'o'
    size = 16
    if lang == 'centroids':
        z = 1
        m = '*'
        size = 128
    if lang == 'custom_centroids':
        z = 1
        m = 'h'
        size = 128
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1],
                c=color, label=lang, zorder=z, marker=m, s=size)


plt.title('encoder_70_mahalanobis_ite1008')
plt.legend()
plt.show()
