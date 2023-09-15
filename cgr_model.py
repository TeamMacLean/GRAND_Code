import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, activations
from os.path import exists
from os import mkdir
import random
import pandas as pd
import numpy as np
from keras.layers import *
from datetime import datetime


# Function to increase, decrease or maintain a number of filters based on the desired shape parameter
def get_filter_list(starting_number, shape, number_of_blocks):
    num_filters = [starting_number]
    for i in range(1, number_of_blocks):
        if shape == 0:
            num_filters.append(num_filters[i - 1])
        elif shape == 1:
            num_filters.append(int(num_filters[i - 1] * 2))
        else:
            num_filters.append(int(num_filters[i - 1] / 2))
    return num_filters


_PRE_ACTIVATION = 'pre_activation'
_START_NUM_FILTERS = 'start_num_filters'
_NUM_FILTER_PATTERN = 'num_filter_pattern'
_FILTER_WIDTH = 'filter_width'
_NUM_CONV_LAYERS = 'num_conv_layers'
_NUM_CONV_BLOCKS = 'num_conv_blocks'
_CONV_DROPOUT = 'conv_dropout'
_USE_BIAS = 'use_bias'
_NUM_DENSE_LAYERS = 'num_dense_layers'
_START_DENSE_LAYERS = 'start_dense_layers'
_DENSE_LAYER_PATTERN = 'dense_layer_pattern'
_DROPOUT = 'dropout'
_DENSE_USE_BIAS = 'dense_use_bias'
_POST_START_NUM_FILTERS = 'post_start_num_filters'
_POST_NUM_FILTER_PATTERN = 'post_num_filter_pattern'
_POST_FILTER_WIDTH = 'post_filter_width'
_POST_NUM_CONV_LAYERS = 'post_num_conv_layers'
_POST_NUM_CONV_BLOCKS = 'post_num_conv_blocks'
_POST_CONV_DROPOUT = 'post_conv_dropout'
_POST_USE_BIAS = 'post_use_bias'
_POOLING_METHOD = 'pooling_method'
_LEARNING_RATE = 'learning_rate'
_OPTIMIZER = 'optimizer'
_REG_L1 = 'reg_l1'
_REG_L2 = 'reg_l2'

POSSIBLE_PARAMS = {
    _PRE_ACTIVATION: [0, 1],
    _START_NUM_FILTERS: [4, 8, 16, 24, 32, 64, 128, 256],
    _NUM_FILTER_PATTERN: [0, 1, 2],
    _FILTER_WIDTH: [3, 5, 7, 9, 11],
    _NUM_CONV_LAYERS: [1, 2, 3],
    _NUM_CONV_BLOCKS: [0, 1, 2, 3],
    _CONV_DROPOUT: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    _USE_BIAS: [0, 1],
    _NUM_DENSE_LAYERS: [0, 1, 2, 3],
    _START_DENSE_LAYERS: [4, 8, 16, 24, 32, 64, 128, 256],
    _DENSE_LAYER_PATTERN: [0, 1, 2],
    _DROPOUT: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    _DENSE_USE_BIAS: [0, 1],
    _POST_START_NUM_FILTERS: [4, 8, 16, 24, 32, 64, 128, 256],
    _POST_NUM_FILTER_PATTERN: [0, 1, 2],
    _POST_FILTER_WIDTH: [3, 5, 7],
    _POST_NUM_CONV_LAYERS: [1, 2, 3],
    _POST_NUM_CONV_BLOCKS: [0, 1],
    _POST_CONV_DROPOUT: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    _POST_USE_BIAS: [0, 1],
    _POOLING_METHOD: [0, 2],
    _LEARNING_RATE: [0.01, 0.001, 0.0015],
    _OPTIMIZER: [0, 1, 2, 3, 4, 5, 6],
    _REG_L1: [0, 1e-05, 0.0001, 0.001, 0.01],
    _REG_L2: [0, 1e-05, 0.0001, 0.001, 0.01]
}

POSSIBLE_PARAMS_SOLO = {
    _PRE_ACTIVATION: [0, 1],
    _START_NUM_FILTERS: [4, 8, 16, 24, 32, 64, 128, 256],
    _NUM_FILTER_PATTERN: [0, 1, 2],
    _FILTER_WIDTH: [3, 5, 7, 9, 11],
    _NUM_CONV_LAYERS: [1, 2, 3],
    _NUM_CONV_BLOCKS: [1, 2, 3],
    _CONV_DROPOUT: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    _USE_BIAS: [0, 1],
    _NUM_DENSE_LAYERS: [0, 1, 2, 3],
    _START_DENSE_LAYERS: [4, 8, 16, 24, 32, 64, 128, 256],
    _DENSE_LAYER_PATTERN: [0, 1, 2],
    _DROPOUT: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    _DENSE_USE_BIAS: [0, 1],
    _POOLING_METHOD: [0, 2],
    _LEARNING_RATE: [1e-2, 1e-3, 15e-4],
    _OPTIMIZER: list(range(7)),
    _REG_L1: [0, 1e-5, 1e-4, 1e-3, 1e-2],
    _REG_L2: [0, 1e-5, 1e-4, 1e-3, 1e-2]}


def get_param_set(i):
    # set all random seeds to i
    np.random.seed(i)
    random.seed(i)
    tf.random.set_seed(i)
    # POSSIBLE_PARAMS is a dictionary of parameters that dictate how the PPI prediction model architecture
    # is built. Each key is a parameter ID, and each value is a list of possible values for the key parameter.
    params = {}
    for key in POSSIBLE_PARAMS:
        params[key] = np.random.choice(POSSIBLE_PARAMS[key])
    return params


# Twinned classifier model. Takes in 2-channel input and sends each channel through a (shared) feature extractor model,
#   passed in as feature_extractor, then concatenates the outputs and passes them through a set of classifier layers.
#   Version of the twinned classifier model that uses Keras' Functional API.
class TwinModel:
    def __init__(self, params, input_shape):
        inputs1 = keras.Input(shape=input_shape[0:2] + (1,), name="cgr1")
        inputs2 = keras.Input(shape=input_shape[0:2] + (1,), name="cgr2")
        lr = tf.cast(params[_LEARNING_RATE], dtype=tf.float64)
        reg_l1 = float(params[_REG_L1])
        reg_l2 = float(params[_REG_L2])
        print(lr, reg_l1, reg_l2)
        # Build ResNet feature extractor
        if bool(params[_PRE_ACTIVATION]):
            first_block_layer = BlockLayerFirstPreActivation
            other_block_layer = BlockLayerPreActivation
        else:
            first_block_layer = BlockLayerFirst
            other_block_layer = BlockLayer
        num_filters = get_filter_list(int(params[_START_NUM_FILTERS]), int(params[_NUM_FILTER_PATTERN]),
                                      int(params[_NUM_CONV_BLOCKS]))
        if params[_NUM_CONV_BLOCKS] > 0:
            fe_layers = [first_block_layer(num_conv_layers=int(params[_NUM_CONV_LAYERS]),
                                           filter_width=int(params[_FILTER_WIDTH]),
                                           num_filters=num_filters[0], use_bias=int(params[_USE_BIAS]),
                                           dropout=params[_CONV_DROPOUT], reg_l1=reg_l1, reg_l2=reg_l2)]
            for i in range(1, int(params[_NUM_CONV_BLOCKS])):
                skip_conv = num_filters[i] != num_filters[i - 1]
                fe_layers.append(other_block_layer(num_conv_layers=int(params[_NUM_CONV_LAYERS]),
                                                   filter_width=int(params[_FILTER_WIDTH]),
                                                   num_filters=num_filters[i], conv_skip=skip_conv,
                                                   use_bias=int(params[_USE_BIAS]),
                                                   dropout=params[_CONV_DROPOUT], reg_l1=reg_l1, reg_l2=reg_l2))
            feature_extractor = keras.Sequential(fe_layers)
            x_1 = feature_extractor(inputs1)
            x_2 = feature_extractor(inputs2)
        else:
            x_1 = inputs1
            x_2 = inputs2
        pooling_method = int(params[_POOLING_METHOD])
        if pooling_method == 0:
            pool = layers.Flatten
        elif pooling_method == 1:
            pool = layers.GlobalMaxPooling2D
        else:
            pool = layers.GlobalAveragePooling2D
        # Concatenate outputs
        if int(params[_POST_NUM_CONV_BLOCKS]) == 0:
            x = layers.concatenate([pool()(x_1), pool()(x_2)])
        else:
            x = layers.concatenate([x_1, x_2])
            post_num_filters = get_filter_list(int(params[_POST_START_NUM_FILTERS]),
                                               int(params[_POST_NUM_FILTER_PATTERN]),
                                               int(params[_POST_NUM_CONV_BLOCKS]))
            # Build ResNet classification
            rc_layers = [first_block_layer(num_conv_layers=int(params[_POST_NUM_CONV_LAYERS]),
                                           filter_width=int(params[_POST_FILTER_WIDTH]),
                                           num_filters=post_num_filters[0], use_bias=int(params[_POST_USE_BIAS]),
                                           dropout=params[_POST_CONV_DROPOUT],
                                           reg_l1=reg_l1, reg_l2=reg_l2)]
            for i in range(1, int(params[_POST_NUM_CONV_BLOCKS])):
                skip_conv = post_num_filters[i] != post_num_filters[i - 1]
                rc_layers.append(other_block_layer(num_conv_layers=int(params[_POST_NUM_CONV_LAYERS]),
                                                   filter_width=int(params[_POST_FILTER_WIDTH]),
                                                   num_filters=post_num_filters[i], conv_skip=skip_conv,
                                                   use_bias=int(params[_POST_USE_BIAS]),
                                                   dropout=params[_POST_CONV_DROPOUT],
                                                   reg_l1=reg_l1, reg_l2=reg_l2))
            resnet_classification = keras.Sequential(rc_layers)
            x = resnet_classification(x)
            x = pool()(x)
        # Build dense layers
        if int(params[_NUM_DENSE_LAYERS]) > 0:
            dense_layers = get_filter_list(int(params[_START_DENSE_LAYERS]),
                                           int(params[_DENSE_LAYER_PATTERN]),
                                           int(params[_NUM_DENSE_LAYERS]))
            cls_layers = []
            for dense_layer in dense_layers:
                cls_layers.append(
                    ClassifierLayer(dense_layer, params[_DROPOUT], use_bias=int(params[_DENSE_USE_BIAS]),
                                    reg_l1=reg_l1, reg_l2=reg_l2))
            classifier_layers = keras.Sequential(cls_layers)
            x = classifier_layers(x)
        # Final classification layer
        outputs = layers.Dense(1, activation="sigmoid", name="output",
                               kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2))(x)
        optimizers = [keras.optimizers.Adam(learning_rate=lr),
                      keras.optimizers.SGD(learning_rate=lr),
                      keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                      keras.optimizers.Adagrad(learning_rate=lr),
                      keras.optimizers.Adadelta(learning_rate=lr),
                      keras.optimizers.Nadam(learning_rate=lr),
                      keras.optimizers.RMSprop(learning_rate=lr)
                      ]
        self.model = keras.Model([inputs1, inputs2], outputs, name="resnet_test")
        self.model.compile(loss=keras.losses.BinaryCrossentropy(),
                           optimizer=optimizers[int(params[_OPTIMIZER])],
                           metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])


# Basic ResNet block. Some convolutional layers with activation function and
# batch normalisation
class BlockLayer(keras.layers.Layer):
    def get_config(self):
        return super(BlockLayer, self).get_config()

    def __init__(self, num_conv_layers=2, num_filters=9, filter_width=3, conv_skip=False, use_bias=False, dropout=0.1,
                 reg_l1=0, reg_l2=0):
        super(BlockLayer, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(
                layers.Conv2D(num_filters, (filter_width, filter_width), activation="relu", padding="same",
                              use_bias=use_bias, kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2)))
            conv_layers.append(layers.BatchNormalization())
        self.conv_layers = keras.Sequential(conv_layers)
        self.num_layers = num_conv_layers
        if num_conv_layers > 1:
            self.conv_skip = conv_skip
            if conv_skip:
                self.conv_skip_layer = layers.Conv2D(num_filters, (filter_width, filter_width), use_bias=use_bias,
                                                     padding="same",
                                                     activation="relu",
                                                     kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))
            self.add = layers.add
        self.dropout_rate = dropout
        if dropout > 0:
            self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.conv_layers(inputs)
        if self.num_layers > 1:
            if self.conv_skip:
                x_s = self.conv_skip_layer(inputs)
            else:
                x_s = inputs
            x = self.add([x, x_s])
        if self.dropout_rate > 0 and training:
            x = self.dropout(x)
        return x


# Version of the ResNet block layer to be used as the first block in a ResNet system - that is, doesn't add a skip
# but instead performs some pooling, and doesn't pad the first convolutional layer
class BlockLayerFirst(keras.layers.Layer):
    def get_config(self):
        return super(BlockLayerFirst, self).get_config()

    def __init__(self, num_conv_layers=2, num_filters=9, filter_width=3, use_bias=False, dropout=0.1, reg_l1=0,
                 reg_l2=0):
        super(BlockLayerFirst, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(
                layers.Conv2D(num_filters, (filter_width, filter_width), use_bias=use_bias, activation="relu",
                              padding="same",
                              kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2)))
            conv_layers.append(layers.BatchNormalization())
        self.conv_layers = keras.Sequential(conv_layers)
        self.pool = layers.MaxPooling2D(padding='same')
        self.dropout_rate = dropout
        if dropout > 0:
            self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.conv_layers(inputs)
        x = self.pool(x)
        if self.dropout_rate > 0 and training:
            x = self.dropout(x)
        return x


class BlockLayerPreActivation(BlockLayer):
    def get_config(self):
        return super(BlockLayerPreActivation, self).get_config()

    def __init__(self, num_conv_layers=2, num_filters=9, filter_width=3, conv_skip=False, use_bias=False, dropout=0.1,
                 reg_l1=0, reg_l2=0):
        super(BlockLayerPreActivation, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(layers.BatchNormalization())
            conv_layers.append(layers.Activation("relu"))
            conv_layers.append(
                layers.Conv2D(num_filters, (filter_width, filter_width), padding="same", use_bias=use_bias,
                              kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2)))
        self.conv_layers = keras.Sequential(conv_layers)
        self.conv_skip = conv_skip
        self.num_layers = num_conv_layers
        if num_conv_layers > 1:
            if conv_skip:
                self.conv_skip_layer = layers.Conv2D(num_filters, (filter_width, filter_width), use_bias=use_bias,
                                                     padding="same",
                                                     activation="relu",
                                                     kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))
            self.add = layers.add
        self.dropout_rate = dropout
        if dropout > 0:
            self.dropout = layers.Dropout(dropout)


class BlockLayerFirstPreActivation(BlockLayerFirst):
    def get_config(self):
        return super(BlockLayerFirstPreActivation, self).get_config()

    def __init__(self, num_conv_layers=2, num_filters=9, filter_width=3, use_bias=False, dropout=0.1, reg_l1=0,
                 reg_l2=0):
        super(BlockLayerFirstPreActivation, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(layers.BatchNormalization())
            conv_layers.append(layers.Activation("relu"))
            conv_layers.append(
                layers.Conv2D(num_filters, (filter_width, filter_width), use_bias=use_bias, padding="same",
                              kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2)))
        self.conv_layers = keras.Sequential(conv_layers)
        self.pool = layers.MaxPooling2D(padding='same')
        self.dropout_rate = dropout
        if dropout > 0:
            self.dropout = layers.Dropout(dropout)


# Single set of batch norm -> dense -> dropout layers.
class ClassifierLayer(keras.layers.Layer):
    def get_config(self):
        return super(ClassifierLayer, self).get_config()

    def __init__(self, dense, dropout, use_bias, reg_l1=0, reg_l2=0):
        super(ClassifierLayer, self).__init__()
        self.norm = layers.BatchNormalization()
        self.dense = layers.Dense(dense, activation="relu", use_bias=use_bias,
                                  kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2))
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.norm(inputs)
        # x = inputs
        x = self.dense(x)
        if training:
            x = self.dropout(x)
        return x


# Version of the CGR model for input that isn't paired
class SingleModel:
    def __init__(self, params, input_shape, binary):
        inputs1 = keras.Input(shape=input_shape)
        # Build ResNet feature extractor
        if bool(params[_PRE_ACTIVATION]):
            first_block_layer = BlockLayerFirstPreActivation
            other_block_layer = BlockLayerPreActivation
        else:
            first_block_layer = BlockLayerFirst
            other_block_layer = BlockLayer
        num_filters = get_filter_list(params[_START_NUM_FILTERS],
                                      params[_NUM_FILTER_PATTERN],
                                      params[_NUM_CONV_BLOCKS])
        if params[_NUM_CONV_BLOCKS] > 0:
            fe_layers = [first_block_layer(num_conv_layers=params[_NUM_CONV_LAYERS],
                                           filter_width=params[_FILTER_WIDTH],
                                           num_filters=num_filters[0], use_bias=params[_USE_BIAS],
                                           dropout=params[_CONV_DROPOUT], reg_l1=params[_REG_L1],
                                           reg_l2=params[_REG_L2])]
            for i in range(1, params[_NUM_CONV_BLOCKS]):
                skip_conv = num_filters[i] != num_filters[i - 1]
                fe_layers.append(other_block_layer(num_conv_layers=params[_NUM_CONV_LAYERS],
                                                   filter_width=params[_FILTER_WIDTH],
                                                   num_filters=num_filters[i], conv_skip=skip_conv,
                                                   use_bias=params[_USE_BIAS],
                                                   dropout=params[_CONV_DROPOUT], reg_l1=params[_REG_L1],
                                                   reg_l2=params[_REG_L2]))
            feature_extractor = keras.Sequential(fe_layers)
            x = feature_extractor(inputs1)
        else:
            x = inputs1
        pooling_method = params[_POOLING_METHOD]
        if pooling_method == 0:
            pool = layers.Flatten
        elif pooling_method == 1:
            pool = layers.GlobalMaxPooling2D
        else:
            pool = layers.GlobalAveragePooling2D
        x = pool()(x)
        # Build dense layers
        if params[_NUM_DENSE_LAYERS] > 0:
            dense_layers = get_filter_list(params[_START_DENSE_LAYERS],
                                           params[_DENSE_LAYER_PATTERN],
                                           params[_NUM_DENSE_LAYERS])
            cls_layers = []
            for dense_layer in dense_layers:
                cls_layers.append(ClassifierLayer(dense_layer, params[_DROPOUT], use_bias=params[_DENSE_USE_BIAS],
                                                  reg_l1=params[_REG_L1], reg_l2=params[_REG_L2]))
            classifier_layers = keras.Sequential(cls_layers)
            x = classifier_layers(x)
        # Final classification layer
        optimizers = [keras.optimizers.Adam(learning_rate=params[_LEARNING_RATE]),
                      keras.optimizers.SGD(learning_rate=params[_LEARNING_RATE]),
                      keras.optimizers.SGD(learning_rate=params[_LEARNING_RATE], momentum=0.9),
                      keras.optimizers.Adagrad(learning_rate=params[_LEARNING_RATE]),
                      keras.optimizers.Adadelta(learning_rate=params[_LEARNING_RATE]),
                      keras.optimizers.Nadam(learning_rate=params[_LEARNING_RATE]),
                      keras.optimizers.RMSprop(learning_rate=params[_LEARNING_RATE])
                      ]
        if binary:
            outputs = layers.Dense(1, activation="sigmoid", name="output",
                                   kernel_regularizer=regularizers.l1_l2(l1=params[_REG_L1],
                                                                         l2=params[_REG_L2]))(x)
            self.model = keras.Model(inputs1, outputs, name="resnet_test")
            self.model.compile(loss=keras.losses.BinaryCrossentropy(),
                               optimizer=optimizers[int(params[_OPTIMIZER])],
                               metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        else:
            outputs = layers.Dense(2, activation="softmax", name="output",
                                   kernel_regularizer=regularizers.l1_l2(l1=params[_REG_L1],
                                                                         l2=params[_REG_L2]))(x)
            self.model = keras.Model(inputs1, outputs, name="resnet_test")
            self.model.compile(loss=keras.losses.CategoricalCrossentropy(),
                               optimizer=optimizers[int(params[_OPTIMIZER])],
                               metrics=['acc'])


# The CGRs are stored un-normalised as integer storage takes up less disk space. Run through a list
# and normalise each CGR
def scale_cgrs(cgr_list):
    new_list = []
    for i in range(cgr_list.shape[0]):
        cgr_1 = cgr_list[i, :, :, 0]
        cgr_2 = cgr_list[i, :, :, 1]
        cgr_1 = cgr_1 / np.amax(cgr_1)
        cgr_2 = cgr_2 / np.amax(cgr_2)
        md_cgr = np.dstack((cgr_1, cgr_2))
        new_list.append(md_cgr)
    return np.stack(new_list)


# Train a set of parameters
def run_params(k, params, saved_data, output_folder, checkpoint_folder=None, seed_i=None, output_prefix='CGR_Model'):
    training_x = scale_cgrs(saved_data['training_x'])
    validation_x = scale_cgrs(saved_data['validation_x'])
    training_y = saved_data['training_y']
    validation_y = saved_data['validation_y']
    train_x_1, train_x_2 = np.split(training_x, 2, axis=3)
    val_x_1, val_x_2 = np.split(validation_x, 2, axis=3)
    if checkpoint_folder is not None:
        if not exists(checkpoint_folder):
            mkdir(checkpoint_folder)
    if not exists(output_folder):
        mkdir(output_folder)
    # If you are using a parameter set randomly generated, include the seed used for random selection,
    if seed_i is not None:
        out_filename = f'{output_folder}/{output_prefix}_{k}mers_{seed_i}.csv'
    # otherwise, just use the current date and time to uniquely name the output file
    else:
        out_filename = f'{output_folder}/{output_prefix}_{k}mers_{datetime.now().strftime("%d%m%y%H%M%S%f")}.csv'
    if not exists(out_filename):
        print(params)
        results_arr = []
        # 5 repetitions for different seeds - check results are consistent:
        for j in range(5):
            np.random.seed(j)
            random.seed(j)
            tf.random.set_seed(j)
            # Train model
            x = 2 ** k
            model = TwinModel(params, (x, x, 2))
            model.model.summary()
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)]
            if seed_i is not None and checkpoint_folder is not None:
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(monitor='val_acc', mode='max',
                                                                    filepath=f'{checkpoint_folder}/AtGrand_{k}mers_{seed_i}',
                                                                    save_weights_only=True, save_best_only=True))
            out = model.model.fit([train_x_1, train_x_2], training_y, epochs=500,
                                  validation_data=([val_x_1, val_x_2], validation_y),
                                  callbacks=callbacks,
                                  verbose=2)
            # Save validation results
            results_arr.append(
                list(params.values()) + [seed_i, j, out.epoch[-1], np.max(out.history['val_acc'])] + [out.history[x][-1]
                                                                                                      for x in
                                                                                                      out.history])
            results_df = pd.DataFrame(results_arr,
                                      columns=list(params.keys()) + ['seed_i', 'seed_j', 'epoch', 'max_val_acc'] + list(
                                          out.history.keys()))
            results_df.to_csv(out_filename)
            tf.keras.backend.clear_session()


def train_submodels(params, seed_i, data_file, dataset_name, out_checkpoints, num_runs=5):
    saved_data = np.load(data_file)
    train_x = scale_cgrs(saved_data['train_X'])
    train_y = saved_data['train_Y']
    train_x_1, train_x_2 = np.split(train_x, 2, axis=3)
    val_x = scale_cgrs(saved_data['val_X'])
    val_y = saved_data['val_Y']
    val_x_1, val_x_2 = np.split(val_x, 2, axis=3)
    for j in range(num_runs):
        np.random.seed(j)
        random.seed(j)
        tf.random.set_seed(j)
        k = params['k']
        x = 2 ** k
        model = TwinModel(params, (x, x, 2))
        checkpoint_filename = f'{out_checkpoints}/{dataset_name}_{k}mers_{seed_i}_{j}'
        if not exists(checkpoint_filename + '.index'):
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                         tf.keras.callbacks.ModelCheckpoint(monitor='val_acc', mode='max',
                                                            filepath=checkpoint_filename,
                                                            save_weights_only=True, save_best_only=True)]
            out = model.model.fit([train_x_1, train_x_2], train_y, epochs=500,
                                  validation_data=([val_x_1, val_x_2], val_y),
                                  callbacks=callbacks,
                                  verbose=2)
            model.model.load_weights(checkpoint_filename)
            tf.keras.backend.clear_session()

        # Holdout tests for transfer learning


# Holdout test - Load checkpoint, do not retrain
def test_mode_no_retraining(params, seed_i, seed_j, data_file, checkpoint_folder):
    saved_data = np.load(data_file)
    test_x = scale_cgrs(saved_data['test_x'])
    test_x_1, test_x_2 = np.split(test_x, 2, axis=3)
    test_y = saved_data['test_y']
    np.random.seed(seed_j)
    random.seed(seed_j)
    tf.random.set_seed(seed_j)
    # Train model
    x = 2 ** 4
    model = TwinModel(params, (x, x, 2))
    checkpoint_filename = f'{checkpoint_folder}/AtGrand_4mers_{seed_i}'
    model.model.load_weights(checkpoint_filename)
    # Save validation results
    out = model.model.evaluate([test_x_1, test_x_2], test_y)
    data = list(params.values()) + [seed_i] + out
    results_df = pd.DataFrame([data],
                              columns=list(params.keys()) + ['seed_i', 'test_loss', 'test_accuracy', 'test_recall',
                                                             'test_precision'])
    tf.keras.backend.clear_session()
    return results_df


# Holdout test - Use the same parameter set but do not load checkpoint from larger dataset
def test_mode_train_from_scratch(params, seed_i, data_file, checkpoint_folder):
    saved_data = np.load(data_file)
    training_x = scale_cgrs(saved_data['training_x'])
    validation_x = scale_cgrs(saved_data['validation_x'])
    training_y = saved_data['training_y']
    validation_y = saved_data['validation_y']
    train_x_1, train_x_2 = np.split(training_x, 2, axis=3)
    val_x_1, val_x_2 = np.split(validation_x, 2, axis=3)
    test_x = scale_cgrs(saved_data['test_x'])
    test_x_1, test_x_2 = np.split(test_x, 2, axis=3)
    test_y = saved_data['test_y']
    # These are the new checkpoints for saving the model trained on the smaller dataset,
    # not the existing checkpoints from the larger dataset
    if not exists(checkpoint_folder):
        mkdir(checkpoint_folder)
    results = []
    for j in range(5):
        np.random.seed(j)
        random.seed(j)
        tf.random.set_seed(j)
        # Train model
        x = 2 ** 4
        model = TwinModel(params, (x, x, 2))
        checkpoint_filename = f'{checkpoint_folder}/EffkGrand_4mers_{seed_i}'
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                     tf.keras.callbacks.ModelCheckpoint(monitor='val_acc', mode='max', filepath=checkpoint_filename,
                                                        save_weights_only=True, save_best_only=True)]
        model.model.fit([train_x_1, train_x_2], training_y, epochs=500,
                        validation_data=([val_x_1, val_x_2], validation_y),
                        callbacks=callbacks,
                        verbose=2)
        model.model.load_weights(checkpoint_filename)
        # Save validation results
        out = model.model.evaluate([test_x_1, test_x_2], test_y)
        data = list(params.values()) + [seed_i] + out
        results.append(data)
        tf.keras.backend.clear_session()
    results_df = pd.DataFrame(results,
                              columns=list(params.keys()) + ['seed_i', 'test_loss', 'test_accuracy', 'test_recall',
                                                             'test_precision'])
    return results_df


# Holdout test - Load the checkpoint from the larger dataset and
# continue to train on smaller dataset.
# If head_only parameter is true, then only retrain the final classification layer
def test_mode_retraining(params, seed_i, data_file, old_checkpoint_folder, new_checkpoint_folder, head_only=False):
    saved_data = np.load(data_file)
    training_x = scale_cgrs(saved_data['training_x'])
    validation_x = scale_cgrs(saved_data['validation_x'])
    training_y = saved_data['training_y']
    validation_y = saved_data['validation_y']
    train_x_1, train_x_2 = np.split(training_x, 2, axis=3)
    val_x_1, val_x_2 = np.split(validation_x, 2, axis=3)
    test_x = scale_cgrs(saved_data['test_x'])
    test_x_1, test_x_2 = np.split(test_x, 2, axis=3)
    test_y = saved_data['test_y']
    results = []
    if not exists(new_checkpoint_folder):
        mkdir(new_checkpoint_folder)
    for j in range(5):
        np.random.seed(j)
        random.seed(j)
        tf.random.set_seed(j)
        # Train model
        x = 2 ** 4
        model = TwinModel(params, (x, x, 2))
        old_checkpoint_filename = f'{old_checkpoint_folder}/AtGrand_4mers_{seed_i}'
        new_checkpoint_filename = f'{new_checkpoint_folder}/EffkGrand_retraining_4mers_{seed_i}'
        model.model.load_weights(old_checkpoint_filename)
        if head_only:
            for i in range(len(model.model.layers) - 1):
                model.model.layers[i].trainable = False
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                     tf.keras.callbacks.ModelCheckpoint(monitor='val_acc', mode='max', filepath=new_checkpoint_filename,
                                                        save_weights_only=True, save_best_only=True)]
        model.model.fit([train_x_1, train_x_2], training_y, epochs=500,
                        validation_data=([val_x_1, val_x_2], validation_y),
                        callbacks=callbacks,
                        verbose=2)
        model.model.load_weights(new_checkpoint_filename)
        # Save validation results
        out = model.model.evaluate([test_x_1, test_x_2], test_y)
        data = list(params.values()) + [seed_i] + out
        results.append(data)
        tf.keras.backend.clear_session()
    results_df = pd.DataFrame(results,
                              columns=list(params.keys()) + ['seed_i', 'test_loss', 'test_accuracy', 'test_recall',
                                                             'test_precision'])
    return results_df


# General inference function - load a checkpoint and run it on some data (with known labels)
def get_individual_predictions(params, seed_i, seed_j, data_file, dataset, checkpoint_filename):
    saved_data = np.load(data_file)
    if dataset + '_x' in saved_data:
        data_x = scale_cgrs(saved_data[dataset + '_x'])
        data_y = saved_data[dataset + '_y']
    else:
        data_x = scale_cgrs(saved_data[dataset + '_X'])
        data_y = saved_data[dataset + '_Y']
    data_x_1, data_x_2 = np.split(data_x, 2, axis=3)
    np.random.seed(seed_j)
    random.seed(seed_j)
    tf.random.set_seed(seed_j)
    x = 2 ** params['k']
    model = TwinModel(params, (x, x, 2))
    model.model.load_weights(checkpoint_filename)
    out_eval = model.model.predict([data_x_1, data_x_2])
    np.random.seed(seed_j)
    random.seed(seed_j)
    tf.random.set_seed(seed_j)
    out = model.model.evaluate([data_x_1, data_x_2], data_y)
    data = list(params.values()) + [seed_i] + out
    results_df = pd.DataFrame([data],
                              columns=list(params.keys()) + ['seed_i', 'test_loss', 'test_accuracy', 'test_recall',
                                                             'test_precision'])
    results_df['test_f1'] = 2 * ((results_df['test_precision'] * results_df['test_recall']) / (
            results_df['test_precision'] + results_df['test_recall']))
    tf.keras.backend.clear_session()
    return out_eval, results_df


# General inference function - load a checkpoint and run it on a query pair (without known labels)
def predict(params, seed_i, seed_j, checkpoint_filename, cgr_1, cgr_2):
    np.random.seed(seed_j)
    random.seed(seed_j)
    tf.random.set_seed(seed_j)
    x = 2 ** params['k']
    model = TwinModel(params, (x, x, 2))
    model.model.load_weights(checkpoint_filename)
    out_eval = model.model.predict([cgr_1[None, :, :], cgr_2[None, :, :]])
    return out_eval
