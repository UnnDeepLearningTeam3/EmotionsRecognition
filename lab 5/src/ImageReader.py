import glob
import mxnet as mx
import cv2
import numpy as np

emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def load_pictures(path):
    pictures = []
    labels = []
    for label_idx, emotion in enumerate(emotions):
        pics = [pic for pic in glob.glob(path + '%s/*' % emotion)]
        for pic in pics[:100]:#pics:#np.random.choice(pics, len(pics)):
            img = cv2.imread(pic)
            img = img.astype('float32') / 255
            pictures.append(img)
            labels.append(label_idx)
    return np.array(pictures), np.array(labels)


def split_data(X, y, percent):
    n = len(X)
    rand_indicies = np.arange(n)
    np.random.shuffle(rand_indicies)
    X = X[rand_indicies]
    y = y[rand_indicies]
    index = int(n * percent / 100)
    return mx.nd.array(X[:index]), mx.nd.array(X[index:]), y[:index], y[index:]
