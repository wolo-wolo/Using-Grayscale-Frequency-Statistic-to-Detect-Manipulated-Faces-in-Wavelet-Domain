# -*- coding: UTF-8 -*-
"""
Record the paths of the four subband images after the Haar wavelet transform into
the training set, validation set, and test set
"""

import os


base_dir_WT = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    base_dir_WT[f] = '../dwt_datasets/DeepfakeTIMIT_faces/' + f

train_dir_WT = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
validation_dir_WT = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
test_dir_WT = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    train_dir_WT[f] = os.path.join(base_dir_WT[f], 'train')
    validation_dir_WT[f] = os.path.join(base_dir_WT[f], 'vadliation')
    test_dir_WT[f] = os.path.join(base_dir_WT[f], 'test')

train_real_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
train_fake_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
validation_real_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
validation_fake_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
test_real_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
test_fake_dir = {'cA/': None, 'cH/': None, 'cV/': None, 'cD/': None}
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    train_real_dir[f] = os.path.join(train_dir_WT[f], 'real')
    train_fake_dir[f] = os.path.join(train_dir_WT[f], 'fake')
    validation_real_dir[f] = os.path.join(validation_dir_WT[f], 'real')
    validation_fake_dir[f] = os.path.join(validation_dir_WT[f], 'fake')
    test_real_dir[f] = os.path.join(test_dir_WT[f], 'real')
    test_fake_dir[f] = os.path.join(test_dir_WT[f], 'fake')