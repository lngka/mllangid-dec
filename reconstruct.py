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


def reconstructFromSTFT(spectrum_phase, logspectrum_stft, save_to):
    abs_spectrum = np.exp(logspectrum_stft)
    spectrum_phase = np.array(spectrum_phase)
    spectrum = abs_spectrum * (np.exp(1j * spectrum_phase))

    istft_graph = tf.Graph()
    with istft_graph.as_default():
        num_fea = int(WIN_SAMPLES / 2 + 1)
        frame_length = WIN_SAMPLES
        frame_step = HOP_SAMPLES

        stft_ph = tf.placeholder(tf.complex64, shape=(None, num_fea))
        samples = tf.signal.inverse_stft(stft_ph, frame_length, frame_step, frame_length, window_fn=tf.signal.inverse_stft_window_fn(
            frame_step, forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)))
        istft_sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True))
        samples_ = istft_sess.run(samples, feed_dict={stft_ph: spectrum})
        wavwrite(save_to, SAMPLING_RATE, samples_)

    return samples_


if __name__ == "__main__":
    pathToPhases = './8K/cn_phases.npy'
    pathToSTFTs = './8K/cn_stfts.npy'

    allPhases = np.load(pathToPhases)
    allSTFTs = np.load(pathToSTFTs)
    print(allSTFTs.shape)
    print(allSTFTs[0, :, :].shape)

    for i in range(100):
        stft = allSTFTs[i, :, :]
        phase = allPhases[i, :, :]
        reconstructFromSTFT(phase, stft, f'./audio/{i}.wav')
