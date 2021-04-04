# -*- coding: UTF-8 -*-


import os, sys
sys.path.append('../')
from dataprepare import dwt_path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from sklearn import metrics
import sys
import json
import numpy as np


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


base_dir = dwt_path.base_dir_WT
train_dir = dwt_path.train_dir_WT
validation_dir = dwt_path.validation_dir_WT
test_dir = dwt_path.test_dir_WT

dataset = 'StyleGAN2'
network = 'Meso4'
# network = 'Xce'
batch_size = 20
input_size = 256

# quantify the mean and std of the GFS differences as the weights
grayFreDiff = '../grayscale frequency statistic and differences/results/diff_mean_std.json'
with open(grayFreDiff, 'r') as f:
    print("Load str file from {}".format(grayFreDiff))
    str1 = f.read()
    r = json.loads(str1)
total_mean = 0
for k, v in r['diff_mean'].items():
    total_mean += abs(v)
weight_mean = {}
for k, v in r['diff_mean'].items():
    weight_mean[k] = abs(v) / total_mean
total_std = 0
for k, v in r['diff_std'].items():
    total_std += v
weight_std = {}
for k, v in r['diff_std'].items():
    weight_std[k] = v / total_std
weight = {}
for k, v in weight_mean.items():
    weight[k] = (v + weight_std[k]) / 2


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


right = 0
labels = []
preds = []
for i in range(len(test_generator_cA)):
    x1, y1 = test_generator_cA.next()
    pred_cA = cA_model.predict(x1)
    x2, y2 = test_generator_cH.next()
    pred_cH = cH_model.predict(x2)
    x3, y3 = test_generator_cV.next()
    pred_cV = cV_model.predict(x3)
    x4, y4 = test_generator_cD.next()
    pred_cD = cD_model.predict(x4)
    labels.append(y1[0])

    pred = weight['cA/']*pred_cA + weight['cH/']*pred_cH + weight['cV/']*pred_cV + weight['cD/']*pred_cD
    if pred[0][0] > 0.5:
        pred[0][0] = 1.
    else:
        pred[0][0] = 0.
    preds.append(pred[0][0])

    if pred[0][0] == y1[0]:
        right += 1


print(network+'-weighted:', file=outputfile)
acc = right / len(test_generator_cA)
print('acc:', acc, file=outputfile)
labels = np.array(labels, dtype=np.float32)
preds = np.array(preds, dtype=np.float32)
AUC = metrics.roc_auc_score(labels, preds)
print('auc:', AUC, file=outputfile)

outputfile.close()
