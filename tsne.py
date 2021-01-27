from dataset import get_shuffled_data_set
import numpy as np
import os
from AutoEncoder import AutoEncoder
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

''' Step1: Get nice data
'''
#languages = ['en', 'de', 'cn', 'fr', 'ru']
languages = ['en', 'cn']

data, labels, _, _ = get_shuffled_data_set(languages, split=False)
data = np.expand_dims(data, -1)

''' Step2: Get embedded data
'''
dir_path = os.path.dirname(os.path.realpath(__file__))
autoencoder = AutoEncoder(n_frames=400, fft_bins=40)

#encoder = autoencoder.load_encoder(model_id='62')

autoencoder.autoencoder.load_weights(
    f'{dir_path}/model_checkpoints/ae_62_re_2/weights.998.hdf5')
encoder = autoencoder.get_encoder()

# encoder = autoencoder.load_encoder(path_to_encoder=f'{dir_path}/model_checkpoints/dec_61/trained_encoder_61_ite0')

data = encoder.predict(data)

X = data
y = labels


''' Step3: Plot TSNE 
'''
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

indices = range(len(languages))

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, color, lang in zip(indices, colors, languages):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=color, label=lang)

plt.title('encoder_62re2_pretrained')
plt.legend()
plt.show()
