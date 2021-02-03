from dataset import get_shuffled_data_set, get_data_set
import numpy as np
import os
from AutoEncoder_2 import AutoEncoder
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt

import scipy as sp
import pandas as pd

from scipy.stats import chi2
from sklearn.covariance import MinCovDet


def robust_mahalanobis_method(df):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(
        df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    # Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    # degrees of freedom = number of variables
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


def mahalanobis_method(df):
    # M-Distance
    x_minus_mu = df - np.mean(df)
    cov = np.cov(df.values.T)  # Covariance
    inv_covmat = sp.linalg.inv(cov)  # Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    # Cut-off point
    # degrees of freedom = number of variables
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


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

#encoder = autoencoder.load_encoder(model_id='62')

autoencoder.autoencoder.load_weights(
    f'{dir_path}/model_checkpoints/ae_70/weights.1925.hdf5')

encoder = autoencoder.get_encoder()
encoder.load_weights('model_checkpoints/dec_70/trained_encoder_ite0.h5')

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
        is_inlier = IsolationForest(
            random_state=0, max_features=50, contamination=0.2).fit_predict(lang_x)
        print(is_inlier)

        # is_inlier = LocalOutlierFactor(
        #     n_neighbors=6, contamination=0.2).fit_predict(lang_x)

        # df = pd.DataFrame(lang_x)
        # outlier, md = robust_mahalanobis_method(df)
        # is_inlier = np.ones(lang_x.shape[0], dtype=int)
        # is_inlier[outlier] = 0

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
    centroids = np.load('model_checkpoints/dec_70/centroids_ite0.npy')
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


plt.title('encoder_70_trained_iso_forest')
plt.legend()
plt.show()
