import os
import math
import time
import tensorflow as tf
if int(tf.version.VERSION.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import numpy as np
import functools
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import pickle

SAMPLING_RATE = 8000
WIN_SAMPLES = int(SAMPLING_RATE * 0.025)
HOP_SAMPLES = int(SAMPLING_RATE * 0.010)
N_FRAMES = 400
# FFT_LENGTH = 198  # fft_bins = fft_length // 2 + 1 = 100
FFT_LENGTH = 78  # fft_bins = fft_length // 2 + 1 = 40


def read_wav(in_path):
    rate, samples = wavread(in_path)
    assert rate == SAMPLING_RATE
    assert samples.dtype == 'int16'
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    assert len(samples.shape) == 1

    return samples


def preprocess(mixedpath, use_mel=False):
    tf.compat.v1.enable_eager_execution()

    try:
        mixedsamples = read_wav(mixedpath)
        mixedsamples = mixedsamples / (max(abs(mixedsamples))+0.000001)
        mixedsamples = mixedsamples.astype(np.float32)

        # Cut the end to have an exact number of frames
        if (len(mixedsamples) - WIN_SAMPLES) % HOP_SAMPLES != 0:
            mixedsamples = mixedsamples[:-
                                        ((len(mixedsamples) - WIN_SAMPLES) % HOP_SAMPLES)]

        # print('================================Wave===========================================')
        # print('SAMPLING_RATE: ', SAMPLING_RATE)
        # print('win_samples', WIN_SAMPLES)
        # print('hop_samples', HOP_SAMPLES)
        # print('mixedsamples', mixedsamples.shape)
        # print('Number of frames',
        #       (1 + (len(mixedsamples) - WIN_SAMPLES) / HOP_SAMPLES))

        mix_wav = mixedsamples
        mix_wav = tf.reshape(mix_wav, [-1])

        mix_stft = tf.signal.stft(
            mix_wav, frame_length=WIN_SAMPLES, frame_step=HOP_SAMPLES, fft_length=FFT_LENGTH)
        mix_stft = toFixedNumFrames(mix_stft, N_FRAMES)

        mix_phase = tf.angle(mix_stft)
        mix_spectrum = tf.log(tf.abs(mix_stft) + 0.00001)

        if use_mel == True:
            mel = librosa.feature.melspectrogram(
                y=mixedsamples, sr=SAMPLING_RATE, n_fft=WIN_SAMPLES, hop_length=HOP_SAMPLES, n_mels=40)
            mel = np.transpose(mel)
            mel = toFixedNumFrames(mel, N_FRAMES)
            return np.array(mix_phase), np.array(mel)

        return np.array(mix_phase), np.array(mix_spectrum)
    except EnvironmentError as err:
        print('Error in handle signal', mixedpath)
        print(err)


def toFixedNumFrames(arr, nframe):
    current = int(arr.shape[0])
    if(current == nframe):
        return arr
    if(current > nframe):
        return arr[-nframe:, :]
    else:
        # fill missing frames with data sliced from end of arr
        missing = nframe - current
        slice_arr = arr[-missing:, :arr.shape[1]]
        new_arr = tf.concat([arr, slice_arr], axis=0)

        missing = nframe - int(new_arr.shape[0])
        while missing > 0:
            slice_arr = arr[-missing:, :arr.shape[1]]
            new_arr = tf.concat([new_arr, slice_arr], axis=0)
            missing = nframe - int(new_arr.shape[0])

        return new_arr


def loadWaveFolder(pathToFiles, use_mel=False):
    phases = None
    features = None

    files = librosa.util.find_files(pathToFiles, ext=['wav'])
    files = np.asarray(files)
    for f in files:
        phase, feature = preprocess(f, use_mel=use_mel)
        feature = np.expand_dims(feature, 0)
        phase = np.expand_dims(phase, 0)

        if features is None:
            features = feature
            phases = phase

        else:
            features = np.concatenate((features, feature), axis=0)
            phases = np.concatenate((phases, phase), axis=0)

    return phases, features, files


def get_shuffled_data_set(languages=['en', 'de', 'cn', 'fr', 'ru'], feature_type='stfts', **kwargs):
    dataset, classes, dataset_test, classes_test = get_data_set(
        languages, feature_type=feature_type, **kwargs)

    dataset, classes = shuffle_data_with_label(dataset, classes)
    dataset_test, classes_test = shuffle_data_with_label(
        dataset_test, classes_test)

    return dataset, classes, dataset_test, classes_test


def get_stacked_data_set(languages=['en', 'de', 'cn', 'fr', 'ru'], feature_type='stfts', n_stack=5, **kwargs):
    '''
    Apply stacking on data set
    n_stack=5 means 1 central row, 2 rows below, 2 rows above
    And move the central row 1 step upward then repeat the stacking process
    '''
    dataset, classes, _, _ = get_data_set(
        languages, feature_type=feature_type, **kwargs)

    n_examples = dataset.shape[0]
    n_rows = dataset.shape[2]
    assert n_rows % n_stack == 0

    stacked_dataset = []
    for i in range(n_examples):
        example = dataset[i]
        stacked_example = example[:, 0:n_stack]
        for j in range(1, n_rows-n_stack):
            start = j
            end = j + n_stack
            stack = example[:, start:end]
            stacked_example = np.concatenate([stacked_example, stack], axis=1)
        stacked_dataset.append(stacked_example)

    return dataset, np.array(stacked_dataset), classes


def get_data_set(languages=['en', 'de', 'cn', 'fr', 'ru'], feature_type='stfts', split=False, get_names=False):
    '''
    feature_type: stfts or mel
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    saveFolder = f'{dir_path}/8K_2000'

    dataset = list()
    classes = list()
    dataset_test = list()
    classes_test = list()
    file_names = list()

    for i in range(len(languages)):
        lang = languages[i]
        features = np.load(
            f'{saveFolder}/{lang}_{feature_type}.npy', allow_pickle=True)
        n_samples = features.shape[0]
        labels = np.full(shape=(n_samples, ), fill_value=i)

        names = np.load(
            f'{saveFolder}/{lang}_file_names.npy', allow_pickle=True)
        names = np.expand_dims(names, axis=0)

        if split == True:
            # X_train, X_test, y_train, y_test = train_test_split(
            #    features, labels, test_size=0.2, random_state=1)
            X_train = features[:950, :, :]
            X_test = features[-50:, :, :]
            y_train = labels[:950, ]
            y_test = labels[-50:, ]

        else:
            X_train = features
            y_train = labels
            X_test = []
            y_test = []

        if i == 0:
            dataset = X_train
            classes = y_train
            dataset_test = X_test
            classes_test = y_test
            file_names = names
        else:
            dataset = np.concatenate((dataset, X_train), axis=0)
            classes = np.concatenate((classes, y_train), axis=0)
            dataset_test = np.concatenate((dataset_test, X_test), axis=0)
            classes_test = np.concatenate((classes_test, y_test), axis=0)
            file_names = np.concatenate((file_names, names), axis=0)

    if get_names:
        return dataset, classes, dataset_test, classes_test, file_names
    else:
        return dataset, classes, dataset_test, classes_test


def shuffle_data_with_label(dataset, classes):
    idx = np.random.permutation(dataset.shape[0])
    x = dataset[idx]
    y = classes[idx]

    return np.array(x), np.array(y)


if __name__ == "__main__":
    use_mel = False
    languages = ['en', 'de', 'cn', 'fr', 'ru']
    #languages = ['en', 'de', 'cn']

    dir_path = os.path.dirname(os.path.realpath(__file__))

    saveFolder = f'{dir_path}/8K_2000'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    for i in range(len(languages)):
        lang = languages[i]

        phases, features, file_names = loadWaveFolder(
            f'/Users/nvckhoa/speech/8K_2000/{lang}', use_mel=use_mel)

        print('lang', lang)
        print('features', features.shape)
        print('file_names', file_names.shape)

        if use_mel == True:
            np.save(f'{saveFolder}/{lang}_mel', features)
        else:
            np.save(f'{saveFolder}/{lang}_stfts', features)

        np.save(f'{saveFolder}/{lang}_phases', phases)
        np.save(f'{saveFolder}/{lang}_file_names', file_names)
