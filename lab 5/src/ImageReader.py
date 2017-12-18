import glob

import cv2
import numpy as np

pic_size = 128
emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def load_pictures(path):
    pictures = []
    validation = []
    for folder in emotions:
        pics = [pic for pic in glob.glob(path + '%s/*' % folder)]
        for pic in pics[:10]:#pics:#np.random.choice(pics, len(pics)):
            img = cv2.imread(pic)
            img = img.astype('float32') / 255
            pictures.append(img)
            validation.append(img) #for encoder use input as desired output
    return np.array(pictures).ravel(), np.array(pictures).ravel()


def split_data(X, y, percent):
    n = len(X)
    rand_indicies = np.arange(n)
    np.random.shuffle(rand_indicies)
    X = X[rand_indicies]
    y = y[rand_indicies]
    index = int(n * percent / 100)
    return X[:index], X[index:], y[:index], y[index:]