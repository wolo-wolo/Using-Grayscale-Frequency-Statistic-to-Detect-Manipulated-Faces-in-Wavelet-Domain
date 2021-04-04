# -*- coding: UTF-8 -*-

import os, sys
sys.path.append('../')
from dataprepare import dwt_path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from sklearn import metrics


dataset = 'DFDC'
# network = 'Meso4'
network = 'Xce'
batch_size = 20
input_size = 128


base_dir = dwt_path.base_dir_WT
train_dir = dwt_path.train_dir_WT
validation_dir = dwt_path.validation_dir_WT
test_dir = dwt_path.test_dir_WT

cA_model = load_model('WT_Network/'+network+'_'+dataset+'_cA.h5')
cH_model = load_model('WT_Network/'+network+'_'+dataset+'_cH.h5')
cV_model = load_model('WT_Network/'+network+'_'+dataset+'_cV.h5')
cD_model = load_model('WT_Network/'+network+'_'+dataset+'_cD.h5')


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


import numpy as np
right_cA, right_cH, right_cV, right_cD = 0, 0, 0, 0
labels = []
preds_cA,  preds_cH, preds_cV, preds_cD = [], [], [], []
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

    preds_cA.append(pred_cA[0][0])
    preds_cH.append(pred_cH[0][0])
    preds_cV.append(pred_cV[0][0])
    preds_cD.append(pred_cD[0][0])

print(network+':', file=outputfile)

labels = np.array(labels, dtype=np.float32)

preds_cA = np.array(preds_cA, dtype=np.float32)
AUC_cA = metrics.roc_auc_score(labels, preds_cA)
test_loss_cA, test_acc_cA = cA_model.evaluate_generator(test_generator_cA, steps=len(test_generator_cA))
print('cA test acc and auc:', test_acc_cA, AUC_cA, file=outputfile)

preds_cH = np.array(preds_cH, dtype=np.float32)
AUC_cH = metrics.roc_auc_score(labels, preds_cH)
test_loss_cH, test_acc_cH = cH_model.evaluate_generator(test_generator_cH, steps=len(test_generator_cH))
print('cH test acc and auc:', test_acc_cH, AUC_cH, file=outputfile)

preds_cV = np.array(preds_cV, dtype=np.float32)
AUC_cV = metrics.roc_auc_score(labels, preds_cV)
test_loss_cV, test_acc_cV = cV_model.evaluate_generator(test_generator_cV, steps=len(test_generator_cV))
print('cV test acc and auc:', test_acc_cV, AUC_cV, file=outputfile)

preds_cD = np.array(preds_cD, dtype=np.float32)
AUC_cD = metrics.roc_auc_score(labels, preds_cD)
test_loss_cD, test_acc_cD = cD_model.evaluate_generator(test_generator_cD, steps=len(test_generator_cD))
print('cD test acc and auc:', test_acc_cD, AUC_cD, file=outputfile)
outputfile.close()
