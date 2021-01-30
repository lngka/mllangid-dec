from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dataset import get_shuffled_data_set
import numpy as np
from AutoEncoder import AutoEncoder


MODEL_ID = '70'  # use to name log txt file and save model

''' Step1: Get nice data
'''
languages = ['en', 'cn']
data, classes, data_test, classes_test = get_shuffled_data_set(
    languages,  split=False)
#data = np.expand_dims(data, -1)


''' Step2: Get embedded data
'''
autoencoder = AutoEncoder()
encoder = autoencoder.load_encoder(model_id=MODEL_ID)

data = encoder.predict(data)
X_train, X_test, y_train, y_test = train_test_split(data, classes, stratify=classes,
                                                    random_state=1, train_size=0.8)

''' Step3: MLP Classification
'''
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

clf = MLPClassifier(hidden_layer_sizes=[
                    128, 64, 32], random_state=1, max_iter=200).fit(X_train, y_train)


pred = clf.predict(X_test)
print('Pred:', pred)
print('y: ', y_test)

score = clf.score(X_test, y_test)
print(score)
