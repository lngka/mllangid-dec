from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dataset import get_shuffled_data_set
import numpy as np
from AutoEncoder import AutoEncoder


data, labels = get_shuffled_data_set(languages=['en', 'cn', 'fr', 'ru'])
data = np.expand_dims(data, -1)

autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder()

data = encoder.predict(data)
print(data.shape)


X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels,
                                                    random_state=1, train_size=0.8)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder()


clf = MLPClassifier(hidden_layer_sizes=[
                    50, 5], random_state=1, max_iter=2048).fit(X_train, y_train)


pred = clf.predict(X_test)
print('Pred:', pred)
print('y: ', y_test)

score = clf.score(X_test, y_test)
print(score)
