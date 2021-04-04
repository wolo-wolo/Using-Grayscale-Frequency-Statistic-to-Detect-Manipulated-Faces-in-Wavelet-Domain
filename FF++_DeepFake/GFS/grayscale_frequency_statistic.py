# -*- coding: UTF-8 -*-
"""
Obtain the grayscale frequency statistics (GFS) and
grayscale frequency differences (GFS differences) of the four subband images after wavelet decomposition
"""


import os, sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from dataprepare import dwt_path


dataset = 'FaceForensics++_DeepFake'
save_path = './results/' # Record data and drawn graphs
if not os.path.exists(save_path):
    os.mkdir(save_path)

gray_real = {}
gray_fake = {}
real_nums = {}
fake_nums = {}

# just use the training set
train_real_dir = dwt_path.train_real_dir
train_fake_dir = dwt_path.train_fake_dir


# grayscale frequency statistic (GFS)
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    gray_real[f] = np.zeros([256], dtype=np.float)
    gray_fake[f] = np.zeros([256], dtype=np.float)
    real_nums[f] = 0
    fake_nums[f] = 0

    for subdir, dirs, files in os.walk(train_real_dir[f]):
        for file in files:
            img = cv2.imread(subdir + '/' + file, 0)
            h, w = np.shape(img)
            for row in range(h):
                for col in range(w):
                    pv = img[row, col]
                    gray_real[f][pv] += 1 / (h * w)
            real_nums[f] += 1

    for subdir, dirs, files in os.walk(train_fake_dir[f]):
        for file in files:
            img = cv2.imread(subdir + '/' + file, 0)
            h, w = np.shape(img)
            for row in range(h):
                for col in range(w):
                    pv = img[row, col]
                    gray_fake[f][pv] += 1 / (h * w)
            fake_nums[f] += 1
    print(f + '\'s num：' + str(real_nums[f]))

# plot GFS
i = 0
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    i += 1
    gray_real[f] /= real_nums[f]
    gray_fake[f] /= fake_nums[f]

    plt.subplot(2, 2, i)
    plt.plot(gray_real[f], color='b', label='real')
    plt.plot(gray_fake[f], color='g', label='fake')
    plt.title(dataset+'-'+f)
    plt.xlim([0, 255])
    plt.xticks(np.linspace(0, 255, 4, endpoint=True))
    plt.xlabel('gray value')
    plt.ylabel('pixel freq')
    plt.tight_layout()
    plt.legend()

plt.savefig(fname=save_path + dataset + '-RealVSFake.svg')
plt.show()


# GFS differences
i = 0
gray_diff = {}
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    i += 1
    gray_diff[f] = gray_fake[f] - gray_real[f]

    plt.subplot(2, 2, i)
    plt.plot(gray_diff[f], color='r', label='diff')
    plt.title(dataset+'-'+f)
    plt.xlim([0, 255])
    plt.xticks(np.linspace(0, 255, 4, endpoint=True))
    plt.xlabel('gray value')
    plt.ylabel('pixel freq difference')
    plt.tight_layout()
    plt.legend()
plt.savefig(fname=save_path + dataset + '-RealVSFake-diff.svg')
plt.show()


# Calculate the mean and std of GFS differences for CNNs weighting
diff_mean = {}
diff_std = {}
for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    diff_mean[f] = gray_diff[f].mean()
    diff_std[f] = gray_diff[f].std()

print(diff_mean)
print(diff_std)

rec = {'diff_mean': diff_mean, 'diff_std': diff_std}
jsObj = json.dumps(rec)
rec_file = open(save_path+'diff_mean_std.json', 'w')
rec_file.write(jsObj)
rec_file.close()
