# -*- coding: UTF-8 -*-
"""
Extract frames from the UADFV video dataset
"""


import cv2
import os


ori_path = '../../original_datasets/UADFV/'

print('process start')
sum = 0  # Record the number of frames extracted
for subdir, dirs, files in os.walk(ori_path):
    for file in files:
        if file[-4:] == '.mp4':
            src_video = os.path.join(subdir, file)
            times = 0
            i = 0
            frameFrequency = 1

            outPutDirName = './frames/' + subdir[-4:]
            if not os.path.exists(outPutDirName):
                os.makedirs(outPutDirName)

            frame = cv2.VideoCapture(src_video)
            while True:
                times += 1
                res, image = frame.read()
                if not res:
                    break
                if times % frameFrequency == 0:
                    cv2.imwrite(outPutDirName + '/' + file[:-4] + '_' + str(times) + '.jpg', image)
                    i += 1
                    sum += 1

            print('Frames extraction of %s finished: %d frames' % (subdir[-5:] + file.split('-')[0], i))
            frame.release()


print('%d frames extracted' % sum)


