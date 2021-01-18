from dataset import get_shuffled_data_set
import numpy as np
from AutoEncoder import AutoEncoder
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

''' Step1: Get nice data
'''
languages = ['en', 'de', 'cn', 'fr', 'ru']
data, labels = get_shuffled_data_set(languages)
data = np.expand_dims(data, -1)

''' Step2: Get embedded data
'''
autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder(model_id='61')
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

plt.title('encoder_61')
plt.legend()
plt.show()
