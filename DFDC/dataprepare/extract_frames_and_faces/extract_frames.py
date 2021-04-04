# -*- coding: UTF-8 -*-
"""
Extract frames from the DFDC video dataset
"""


import json
import os
import cv2

path = '../../original_datasets/dfdc/train_sample_videos'
with open(path + '/metadata.json', 'r') as load_f:
    info = json.load(load_f)

FAKE = 0
REAL = 0
for subdir, dirs, files in os.walk(path):
    for file in files:
        if file in info.keys():
            if info[file]['label'] == 'FAKE':
                src_video = os.path.join(subdir, file)
                times = 0
                i = 0
                frameFrequency = 4

                outPutDirName = './frames/' + 'FAKE' + '/'

                if not os.path.exists(outPutDirName):
                    os.makedirs(outPutDirName)

                frame = cv2.VideoCapture(src_video)
                while True:
                    times += 1
                    res, image = frame.read()
                    if not res:
                        break
                    if times % frameFrequency == 0:
                        cv2.imwrite(outPutDirName + file + '%' + str(times) + '.jpg', image)
                        i += 1
                        FAKE += 1
                frame.release()

            if info[file]['label'] == 'REAL':
                src_video = os.path.join(subdir, file)

                times = 0
                i = 0
                frameFrequency = 1

                outPutDirName = './frames/' + 'REAL' + '/'
                if not os.path.exists(outPutDirName):
                    os.makedirs(outPutDirName)

                frame = cv2.VideoCapture(src_video)
                while True:
                    times += 1
                    res, image = frame.read()
                    if not res:
                        break
                    if times % frameFrequency == 0:
                        cv2.imwrite(outPutDirName + file + '%' + str(times) + '.jpg', image)
                        i += 1
                        REAL += 1
                frame.release()


print('%d fake frames extracted' % FAKE)
print('%d real frames extracted' % REAL)
