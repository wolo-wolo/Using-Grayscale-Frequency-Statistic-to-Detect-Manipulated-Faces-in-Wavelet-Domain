# -*- coding: UTF-8 -*-

import os, sys
sys.path.append('../')
from dataprepare import dwt_path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
from keras import backend as K


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


dataset = 'StyleGAN2'
network = 'Meso4'
# network = 'Xce'
batch_size = 20
input_size = 256


base_dir = dwt_path.base_dir_WT
train_dir = dwt_path.train_dir_WT
validation_dir = dwt_path.validation_dir_WT
test_dir = dwt_path.test_dir_WT

cA_model = load_model('WT_Network/'+network+'_'+dataset+'_cA.h5', custom_objects={'auc': auc})
cH_model = load_model('WT_Network/'+network+'_'+dataset+'_cH.h5', custom_objects={'auc': auc})
cV_model = load_model('WT_Network/'+network+'_'+dataset+'_cV.h5', custom_objects={'auc': auc})
cD_model = load_model('WT_Network/'+network+'_'+dataset+'_cD.h5', custom_objects={'auc': auc})


test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator_cA = test_datagen.flow_from_directory(test_dir['cA/'],
                                                     target_size=(input_size, input_size),
                                                     batch_size=batch_size,
                                                     class_mode='binary',
                                                     shuffle=False)
test_generator_cH = test_datagen.flow_from_directory(test_dir['cH/'],
                                                     target_size=(input_size, input_size),
                                                     batch_size=batch_size,
                                                     class_mode='binary',
                                                     shuffle=False)
test_generator_cV = test_datagen.flow_from_directory(test_dir['cV/'],
                                                     target_size=(input_size, input_size),
                                                     batch_size=batch_size,
                                                     class_mode='binary',
                                                     shuffle=False)
test_generator_cD = test_datagen.flow_from_directory(test_dir['cD/'],
                                                     target_size=(input_size, input_size),
                                                     batch_size=batch_size,
                                                     class_mode='binary',
                                                     shuffle=False)


filename = 'result'
output = sys.stdout
outputfile = open("./" + filename + '.txt', 'a+')
sys.stdout = outputfile


print(network+':', file=outputfile)
test_loss_cA, test_acc_cA, test_auc_cA = cA_model.evaluate_generator(test_generator_cA, steps=len(test_generator_cA))
print('cA test acc and auc:', test_acc_cA, test_auc_cA, file=outputfile)
test_loss_cH, test_acc_cH, test_auc_cH = cH_model.evaluate_generator(test_generator_cH, steps=len(test_generator_cH))
print('cH test acc and auc:', test_acc_cH, test_auc_cH, file=outputfile)
test_loss_cV, test_acc_cV, test_auc_cV = cV_model.evaluate_generator(test_generator_cV, steps=len(test_generator_cV))
print('cV test acc and auc:', test_acc_cV, test_auc_cV, file=outputfile)
test_loss_cD, test_acc_cD, test_auc_cD = cD_model.evaluate_generator(test_generator_cD, steps=len(test_generator_cD))
print('cD test acc and auc:', test_acc_cD, test_auc_cD, file=outputfile)
outputfile.close()
