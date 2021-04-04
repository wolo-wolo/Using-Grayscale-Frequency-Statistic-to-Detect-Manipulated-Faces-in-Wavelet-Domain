# -*- coding: UTF-8 -*-
"""
Extract frames from the CelebDF_V2 video dataset
"""

import cv2
import os

ori_path = ['../../original_datasets/Celeb-real',
            '../../original_datasets/Celeb-synthesis',
            '../../original_datasets/YouTube-real']
labels = [0, 1, 0]
print('process start')

total = 0
for z in range(len(ori_path)):
    sum = 0
    if labels[z] == 0:
        for subdir, dirs, files in os.walk(ori_path[z]):
            for file in files:
                if file[-4:] == '.mp4':
                    src_video = os.path.join(subdir, file)

                    times = 0
                    i = 0
                    frameFrequency = 5

                    outPutDirName = './frames/' + ori_path[z].split('/')[-1] + '/'
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
                            sum += 1
                            total += 1
                    frame.release()

            print('Frames extraction of %s finished: %d frames' % (subdir[subdir.find('2') + 2:], sum))

    if labels[z] == 1:
        for subdir, dirs, files in os.walk(ori_path[z]):
            for file in files:
                if file[-4:] == '.mp4':
                    src_video = os.path.join(subdir, file)
                    times = 0
                    i = 0
                    frameFrequency = 30

                    outPutDirName = './frames/' + ori_path[z].split('/')[-1] + '/'
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
                            sum += 1
                            total += 1
                    frame.release()

            print('Frames extraction of %s finished: %d frames' % (subdir[subdir.find('2') + 2:], sum))

print('%d frames extracted' % total)

