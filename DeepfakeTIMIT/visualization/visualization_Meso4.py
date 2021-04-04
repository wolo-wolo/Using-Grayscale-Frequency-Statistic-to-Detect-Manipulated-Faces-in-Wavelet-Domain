# -*- coding: UTF-8 -*-

import sys

sys.path.append('../')
from dataprepare import dwt_path
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import os
import cv2
import random


save_dir = './CAM_heatmap/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for wt_coe in ['cA', 'cH', 'cV', 'cD']:
    model = load_model('../train_and_test/WT_Network/Meso4_DeepfakeTIMIT_' + wt_coe + '.h5')
    img_source = dwt_path.train_fake_dir[wt_coe+'/']


    i = 0
    for subdir, dirs, files in os.walk(img_source):
        pathDir = os.listdir(os.path.join(subdir))
        img_number = len(pathDir)
        rate = 10 / img_number
        picknumber = int(img_number * rate)
        random.seed(0)
        sample = random.sample(pathDir, picknumber)
        print(sample)
        for file in sample:
            if file[-4:] == '.jpg':
                img_path = os.path.join(subdir, file)

                img = image.load_img(img_path, target_size=(112, 112))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # pred
                preds = model.predict(x)
                print('Predicted:', preds)
                print('Predicted:', preds[0])
                print(preds.shape)
                print(np.argmax(preds[0]))

                # Grad-CAN
                predicted_fake_output = model.output[:, np.argmax(preds[0])]

                wt_coe_layer = {'cA': '4', 'cH': '8', 'cV': '12', 'cD': '16'}

                last_conv_layer = model.get_layer('conv2d_' + wt_coe_layer[wt_coe])  # the last convolution layerr
                grads = K.gradients(predicted_fake_output, last_conv_layer.output)[0]

                pooled_grad = K.mean(grads, axis=(0, 1, 2))
                iterate = K.function([model.input], [pooled_grad, last_conv_layer.output[0]])
                pooled_grads_value, conv_layer_output_value = iterate([x])

                for i in range(8):
                    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
                heatmap = np.mean(conv_layer_output_value, axis=-1)

                # heat map post-process
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                img = cv2.imread(img_path)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.7 + img
                cv2.imwrite('./CAM_heatmap/' + wt_coe + file, superimposed_img)
