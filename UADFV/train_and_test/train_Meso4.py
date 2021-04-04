# -*- coding: UTF-8 -*-
"""
Train four Meso-4 networks separately for the four subband images after wavelet decomposition.
We save the weight of the model with the lowest validation loss.
Our experiment is performed on an NVIDIA GeForce RTX 2080Ti GPU.
"""


import os, sys
sys.path.append('../')
from dataprepare import dwt_path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from numba import cuda


# datasets path
base_dir = dwt_path.base_dir_WT
train_dir = dwt_path.train_dir_WT
validation_dir = dwt_path.validation_dir_WT
test_dir = dwt_path.test_dir_WT

batch_size = 20
input_size = 128
data = 'UADFV'

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

for spec_feature in ['cA/', 'cH/', 'cV/', 'cD/']:
    print('Train on :', spec_feature)
    train_generator = train_datagen.flow_from_directory(train_dir[spec_feature],
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    validation_generator = val_datagen.flow_from_directory(validation_dir[spec_feature],
                                                           target_size=(input_size, input_size),
                                                           batch_size=batch_size,
                                                           class_mode='binary')

    # construct Meso-4 Network
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(input_size, input_size, 3),))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(8, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model compile and fit with images
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])

    network_dir = './WT_Network/'  # save trained networks
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath=network_dir + 'Meso4_' + data + '_' + spec_feature[:-1] + '.h5',
                                        monitor='val_loss',
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ]
    history = model.fit_generator(train_generator,
                                  workers=8,
                                  steps_per_epoch=len(train_generator),
                                  epochs=40,
                                  validation_data=validation_generator,
                                  validation_steps=len(validation_generator),
                                  verbose=2,
                                  callbacks=callbacks_list)

    # plot acc and loss
    train_record_dir = './loss_acc_rec/'
    if not os.path.exists(train_record_dir):
        os.mkdir(train_record_dir)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('train and validation accuracy')
    plt.legend()
    plt.savefig(fname=train_record_dir + 'Meso4_' + data + '_acc_' + spec_feature[:-1] + '.svg')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('train and validation loss')
    plt.legend()
    plt.savefig(fname=train_record_dir + 'Meso4_' + data + '_loss_' + spec_feature[:-1] + '.svg')
    # plt.show()

    K.clear_session()
    # tf.reset_default_graph()
    # cuda.close()

