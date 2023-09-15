import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, activations
from os.path import join, basename, exists
from tensorflow.keras.models import Sequential
from os import mkdir
import pandas as pd
from time import sleep
import sys
from keras import backend as K
from sklearn.datasets import load_digits
import random
import pandas as pd
import numpy as np
from keras.layers import *
from keras import metrics
from datetime import datetime
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sb


# Much simplified version of the CGR CNN model used for taxonomic classification

def get_simple_model(params, input_shape, num_classes):
    model = Sequential()
    for j in range(params['conv_blocks']):
        for i in range(params['num_conv_layers_per_block']):
            if i == 0 and j == 0:
                model.add(Conv2D(filters=params['num_filters'],
                                 kernel_size=(params['kernel_width'], params['kernel_width']),
                                 activation='relu', padding='same', use_bias=params['conv_use_bias'],
                                 input_shape=input_shape
                                 ))
            else:
                model.add(Conv2D(filters=params['num_filters'],
                                 kernel_size=(params['kernel_width'], params['kernel_width']),
                                 activation='relu', padding='same', use_bias=params['conv_use_bias']
                                 ))
        model.add(MaxPool2D((2, 2)))
        if params['rate_of_conv_dropout'] > 0:
            model.add(Dropout(params['rate_of_conv_dropout']))
        if bool(params['conv_batch_normalisation']):
            model.add(BatchNormalization())
    model.add(Flatten())
    for i in range(params['dense_layers']):
        model.add(Dense(params['dense_layer_depth'], activation='relu', use_bias=params['dense_use_bias']))
        if bool(params['dense_batch_normalisation']):
            model.add(BatchNormalization())
    if params['rate_of_dropout'] > 0.0:
        model.add(Dropout(params['rate_of_dropout']))
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        optimizers = [tf.keras.optimizers.Adam(lr=params['learning_rate']),
                      tf.keras.optimizers.SGD(lr=params['learning_rate']),
                      tf.keras.optimizers.SGD(lr=params['learning_rate'], momentum=0.9),
                      tf.keras.optimizers.Adagrad(lr=params['learning_rate'])
                      ]
        model.compile(optimizer=optimizers[int(params['optimizer'])],
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        optimizers = [tf.keras.optimizers.Adam(lr=params['learning_rate']),
                      tf.keras.optimizers.SGD(lr=params['learning_rate']),
                      tf.keras.optimizers.SGD(lr=params['learning_rate'], momentum=0.9),
                      tf.keras.optimizers.Adagrad(lr=params['learning_rate'])
                      ]
        model.compile(optimizer=optimizers[int(params['optimizer'])],
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc', tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.CategoricalCrossentropy()])
    # metrics=['acc',keras.metrics.Precision(),keras.metrics.Recall(), keras.metrics.AUC()])
    return model


def plot_cm(conf, species_list, rotate_x_labels=True):
    # plt.figure(figsize=(6,6))
    row_sums = conf.sum(axis=1)
    conf = conf / row_sums[:, np.newaxis]
    plt.imshow(conf, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if rotate_x_labels:
        plt.xticks(range(len(species_list)), species_list, rotation='vertical')
    else:
        plt.xticks(range(len(species_list)), species_list)
    plt.yticks(range(len(species_list)), species_list)
    plt.colorbar()
    midp = (np.amax(conf) + np.amin(conf)) / 2
    for i in range(len(species_list)):
        for j in range(len(species_list)):
            if conf[i][j] > midp:
                colour = 'w'
            else:
                colour = 'k'
            plt.text(j, i, '{:.2f}'.format(conf[i][j]), c=colour)
    plt.show()
    accs = conf.diagonal()
    print(len(accs))
    print('mean class accuracy', np.mean(accs))


def test_params(params, species_list, seed_j, train_x, train_y, val_x, val_y, test_x, test_y, class_weights):
    model = get_simple_model(params, train_x.shape[1:], len(species_list))
    np.random.seed(seed_j)
    random.seed(seed_j)
    tf.random.set_seed(seed_j)
    model.fit(train_x, train_y, epochs=500, validation_data=(val_x, val_y),
              callbacks=[EarlyStopping(patience=5, monitor='val_loss')],
              class_weight=class_weights, batch_size=128, verbose=1)
    pred_y = model.predict(test_x)
    cm = confusion_matrix(np.argmax(test_y, axis=1), np.argmax(pred_y, axis=1), labels=range(len(species_list)))
    plot_cm(cm, species_list)
    return pred_y, cm


def validate_params(params, species_list, train_x, train_y, val_x, val_y, class_weights):
    model = get_simple_model(params, train_x.shape[1:], len(species_list))
    out = model.fit(train_x, train_y, epochs=250, validation_data=(val_x, val_y),
                    callbacks=[EarlyStopping(patience=5, monitor='val_loss')],
                    class_weight=class_weights, batch_size=128, verbose=1)
    return out


possible_params = {
    'conv_blocks': [0, 1, 2, 3, 4],
    'num_filters': [16, 24, 32, 64, 128, 256, 512, 1024],
    'num_conv_layers_per_block': [1, 2, 3, 4],
    'kernel_width': [3, 5, 7, 9, 11],
    'dense_layer_depth': [16, 24, 32, 64, 128, 256, 512, 1024],
    'dense_layers': [0, 1, 2, 3, 4],
    'conv_batch_normalisation': [0, 1],
    'dense_batch_normalisation': [0, 1],
    'rate_of_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'rate_of_conv_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'conv_use_bias': [0, 1],
    'dense_use_bias': [0, 1],
    'learning_rate': [1e-2, 1e-3, 15e-4],
    'optimizer': list(range(4))}
