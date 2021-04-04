# -*- coding: UTF-8 -*-
"""
Extract frames from the FF++_Face2Face video dataset
"""

import cv2
import os


ori_path = ['../../original_datasets/youtube/c40/videos',
            '../../original_datasets/Face2Face/c40/videos']
labels = [0, 1]
print('process start')


# Get the total number of frames of true and fake videos respectively, for balancing the exacted frames
frameFrequenceBalance = []
for z in range(len(ori_path)):
    sum = 0  # Record the number of frames extracted
    for subdir, dirs, files in os.walk(ori_path[z]):
        for file in files:
            if file[-4:] == '.mp4':
                src_video = os.path.join(subdir, file)
                frame = cv2.VideoCapture(src_video)
                frame_nums = frame.get(7)
                sum += frame_nums
        frameFrequenceBalance.append(sum)
extract_frames = 30000  # Control the final number of frames extracted from each type (real or fake) of videos


total = 0  # Record the number of frames extracted
for z in range(len(ori_path)):

    if labels[z] == 0:
        sum = 0
        for subdir, dirs, files in os.walk(ori_path[z]):
            for file in files:
                if file[-4:] == '.mp4':
                    src_video = os.path.join(subdir, file)

                    times = 0
                    i = 0
                    frameFrequency = int(frameFrequenceBalance[z]/30000)

                    outPutDirName = './frames/' + ori_path[z].split('/')[3] + '/'
                    if not os.path.exists(outPutDirName):
                        os.makedirs(outPutDirName)

                    frame = cv2.VideoCapture(src_video)
                    while True:
                        times += 1
                        res, image = frame.read()
                        if not res:
                            break
                        if times % frameFrequency == 0:
                            cv2.imwrite(outPutDirName + subdir.split('/')[3] + file + '%' + str(times) + '.jpg', image)
                            i += 1
                            sum += 1
                            total += 1
                    frame.release()

            print('Frames extraction of %s finished: %d frames' % (subdir.split('/')[3], sum))

    if labels[z] == 1:
        sum = 0
        for subdir, dirs, files in os.walk(ori_path[z]):
            for file in files:
                if file[-4:] == '.mp4':
                    src_video = os.path.join(subdir, file)
                    times = 0
                    i = 0
                    frameFrequency = int(frameFrequenceBalance[z]/30000)

                    outPutDirName = './frames/' + ori_path[z].split('/')[3] + '/'
                    if not os.path.exists(outPutDirName):
                        os.makedirs(outPutDirName)

                    frame = cv2.VideoCapture(src_video)
                    while True:
                        times += 1
                        res, image = frame.read()
                        if not res:
                            break
                        if times % frameFrequency == 0:
                            cv2.imwrite(outPutDirName + subdir.split('/')[3] + file + '%' + str(times) + '.jpg', image)
                            i += 1
                            sum += 1
                            total += 1
                    frame.release()
            print('Frames extraction of %s finished: %d frames' % (subdir.split('/')[3], sum))

print('%d frames extracted' % total)
