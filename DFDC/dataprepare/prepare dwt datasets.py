# -*- coding: UTF-8 -*-
"""
All images in the original dataset are subjected to Haar wavelet transform.
The obtained cA, cH, cV, and cD subband images are divided into training set, validation set, and testing set respectively
"""

import os
import dwt_path
import cv2
import numpy as np
from pywt import dwt2


# Create training dirs, validation dirs and testing dirs for the four subbands after wavelet transform
base_dir = dwt_path.base_dir_WT
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    if not os.path.exists(base_dir[f]):
        print(base_dir[f])
        os.makedirs(base_dir[f])

    train_dir = dwt_path.train_dir_WT
    if not os.path.exists(train_dir[f]):
        os.mkdir(train_dir[f])
    validation_dir = dwt_path.validation_dir_WT
    if not os.path.exists(validation_dir[f]):
        os.mkdir(validation_dir[f])
    test_dir = dwt_path.test_dir_WT
    if not os.path.exists(test_dir[f]):
        os.mkdir(test_dir[f])

    train_real_dir = dwt_path.train_real_dir
    if not os.path.exists(train_real_dir[f]):
        os.mkdir(train_real_dir[f])
    train_fake_dir = dwt_path.train_fake_dir
    if not os.path.exists(train_fake_dir[f]):
        os.mkdir(train_fake_dir[f])

    validation_real_dir = dwt_path.validation_real_dir
    if not os.path.exists(validation_real_dir[f]):
        os.mkdir(validation_real_dir[f])
    validation_fake_dir = dwt_path.validation_fake_dir
    if not os.path.exists(validation_fake_dir[f]):
        os.mkdir(validation_fake_dir[f])

    test_real_dir = dwt_path.test_real_dir
    if not os.path.exists(test_real_dir[f]):
        os.mkdir(test_real_dir[f])
    test_fake_dir = dwt_path.test_fake_dir
    if not os.path.exists(test_fake_dir[f]):
        os.mkdir(test_fake_dir[f])


def haar_dwt(filename, dst, fre=None):
    img = cv2.imread(filename, 0)
    cA, [cH, cV, cD] = dwt2(img, 'haar')
    if fre == 'cA/':
        cv2.imwrite(dst, np.uint8(cA))
    elif fre == 'cH/':
        cv2.imwrite(dst, np.uint8(cH))
    elif fre == 'cV/':
        cv2.imwrite(dst, np.uint8(cV))
    elif fre == 'cD/':
        cv2.imwrite(dst, np.uint8(cD))


# The original DeepfakeTIMIT dataset path
path = ['./extract_frames_and_faces/faces/DFDC_real_faces/',
        './extract_frames_and_faces/faces/DFDC_fake_faces/',
        ]
labels = [0, 1]  # 0 means true, 1 means fake
format_file = ['.jpg', '.jpg']  # corresponding image format


for z in range(len(path)):
    print('process in :', path[z])
    if labels[z] == 0:
        i = 0
        for subdir, dirs, files in os.walk(path[z]):
            for file in files:
                if file[-4:] == format_file[z] and i < len(files) * 0.6:
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(train_real_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)

                elif file[-4:] == format_file[z] and i < len(files) * 0.8:
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(validation_real_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)

                elif file[-4:] == format_file[z] and i < len(files):
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(test_real_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)

                i += 1

    if labels[z] == 1:
        i = 0
        for subdir, dirs, files in os.walk(path[z]):
            for file in files:
                if file[-4:] == format_file[z] and i < 0.6 * len(files):
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(train_fake_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)

                elif file[-4:] == format_file[z] and i < 0.8 * len(files):
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(validation_fake_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)

                elif file[-4:] == format_file[z] and i < len(files):
                    src = os.path.join(subdir, file)
                    dfname = file
                    print(dfname)
                    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
                        dst = os.path.join(test_fake_dir[f], dfname)
                        haar_dwt(src, dst, fre=f)
                i += 1

for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    print('total training real images:', len(os.listdir(train_real_dir[f])))
    print('total training fake images:', len(os.listdir(train_fake_dir[f])))
    print('total validation real images:', len(os.listdir(validation_real_dir[f])))
    print('total validation fake images:', len(os.listdir(validation_fake_dir[f])))
    print('total test real images:', len(os.listdir(test_real_dir[f])))
    print('total test fake images:', len(os.listdir(test_fake_dir[f])))



