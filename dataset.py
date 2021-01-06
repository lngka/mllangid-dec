import os
import math
import time
import tensorflow as tf
if int(tf.version.VERSION.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
import numpy as np
import functools
import librosa
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

SAMPLING_RATE = 8000
WIN_SAMPLES = int(SAMPLING_RATE * 0.025)
HOP_SAMPLES = int(SAMPLING_RATE * 0.010)
N_FRAMES = 400
# FFT_LENGTH = 198  # fft_bins = fft_length // 2 + 1 = 100
FFT_LENGTH = 78  # fft_bins = fft_length // 2 + 1 = 100


def read_wav(in_path):
    rate, samples = wavread(in_path)
    assert rate == SAMPLING_RATE
    assert samples.dtype == 'int16'
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    assert len(samples.shape) == 1

    return samples


def preprocess(mixedpath):
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

        tf.compat.v1.enable_eager_execution()

        # data processing
        mix_wav = mixedsamples
        mix_wav = tf.reshape(mix_wav, [-1])

        # print('================================reshape=================================================')
        # print('mix_wav', mix_wav.shape)

        mix_stft = tf.signal.stft(
            mix_wav, frame_length=WIN_SAMPLES, frame_step=HOP_SAMPLES, fft_length=FFT_LENGTH)

        mix_stft = toFixedNumFrames(mix_stft, N_FRAMES)

        # print('================================STFT=====================================================')
        # print('mix_stft: ', mix_stft.shape)

        mix_phase = tf.angle(mix_stft)
        mix_spectrum = tf.log(tf.abs(mix_stft) + 0.00001)

        return mix_phase, mix_spectrum
    except EnvironmentError as err:
        print('Error in handle signal', mixedpath)
        print(err)


def toFixedNumFrames(arr, nframe):
    current = arr.shape[0]
    if(current == nframe):
        return arr
    if(current > nframe):
        return arr[:nframe, :]
    else:
        # fill missing frames with data sliced from start of arr
        missing = nframe - current
        slice_arr = arr[:missing, :arr.shape[1]]
        new_arr = tf.concat([arr, slice_arr], axis=0)

        # fill missing frames with zeros
        #new_arr = np.zeros(shape=[nframe, arr.shape[1]], dtype=complex)
        #new_arr[:arr.shape[0], :arr.shape[1]] = arr
        return new_arr


def loadWaveFolder(pathToFiles):
    phases = []
    stfts = []

    files = librosa.util.find_files(pathToFiles, ext=['wav'])
    files = np.asarray(files)
    for f in files:
        phase, stft = preprocess(f)
        phases.append(np.array(phase))
        stfts.append(np.array(stft))

    return np.array(phases), np.array(stfts)


def get_shuffled_data_set(languages=['en', 'de', 'cn', 'fr', 'ru']):
    #languages = ['en', 'de', 'cn', 'fr', 'ru']

    dir_path = os.path.dirname(os.path.realpath(__file__))
    saveFolder = f'{dir_path}/8K'

    dataset = list()
    classes = list()

    for i in range(len(languages)):
        lang = languages[i]
        stft = np.load(f'{saveFolder}/{lang}_stfts.npy')
        n_samples = stft.shape[0]
        label = np.full(shape=(n_samples, ), fill_value=i)
        #label = np.load(f'{saveFolder}/{lang}_labels.npy')

        if i == 0:
            dataset = stft
            classes = label
        else:
            dataset = np.concatenate((dataset, stft), axis=0)
            classes = np.concatenate((classes, label), axis=0)

    dataset, classes = shuffle_data_with_label(dataset, classes)
    return dataset, classes


def get_data_set(languages=['en', 'de', 'cn', 'fr', 'ru']):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    saveFolder = f'{dir_path}/8K'

    dataset = list()
    classes = list()

    for i in range(len(languages)):
        lang = languages[i]
        stft = np.load(f'{saveFolder}/{lang}_stfts.npy')
        label = np.load(f'{saveFolder}/{lang}_labels.npy')

        if i == 0:
            dataset = stft
            classes = label
        else:
            dataset = np.concatenate((dataset, stft), axis=0)
            classes = np.concatenate((classes, label), axis=0)

    return dataset, classes


def shuffle_data_with_label(dataset, classes):
    idx = np.random.permutation(dataset.shape[0])
    x = dataset[idx]
    y = classes[idx]

    return np.array(x), np.array(y)


if __name__ == "__main__":
    languages = ['en', 'de', 'cn', 'fr', 'ru']
    dir_path = os.path.dirname(os.path.realpath(__file__))

    saveFolder = f'{dir_path}/8K'

    for i in range(len(languages)):
        lang = languages[i]
        phases, stfts = loadWaveFolder(f'/Users/nvckhoa/speech/8K/{lang}')

        labels = np.full(phases.shape[0], i)

        np.save(f'{saveFolder}/{lang}_phases', phases)
        np.save(f'{saveFolder}/{lang}_stfts', stfts)
        np.save(f'{saveFolder}/{lang}_labels', labels)

        print(lang)
        print(phases.shape)
        print(stfts.shape)
        print(labels.shape)
        print(labels)
