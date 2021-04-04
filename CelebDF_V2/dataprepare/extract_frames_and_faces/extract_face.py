# -*- coding: UTF-8 -*-


import dlib
import numpy as np
import cv2
import os

face_dir = ['./faces/Celeb-real/',
            './faces/Celeb-synthesis/',
            './faces/YouTube-real/']


for i in range(len(face_dir)):
    if not os.path.exists(face_dir[i]):
        os.makedirs(face_dir[i])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def crop_face(src, dst):
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    dets = detector(img, 1)
    print("face number：", len(dets))
    for k, d in enumerate(dets):
        if len(dets) > 1:
            continue
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            if d.top() + i >= img_height:
                continue
            for j in range(width):
                if d.left() + j >= img_width:
                    continue
                img_blank[i][j] = img[d.top() + i][d.left() + j]
        img_blank = cv2.resize(img_blank, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imencode('.jpg', img_blank)[1].tofile(dst)


frames_path = ['./frames/Celeb-real/',
               './frames/Celeb-synthesis/',
               './frames/YouTube-real/']
labels = [0, 1, 0]
format_file = ['.jpg', '.jpg', '.jpg']

real_face_num = 0
fake_face_num = 0
for z in range(len(frames_path)):
    print('process in :', frames_path[z])
    if labels[z] == 0:
        for subdir, dirs, files in os.walk(frames_path[z]):
            for file in files:
                if file[-4:] == format_file[z]:
                    src = os.path.join(subdir, file)
                    dfname = file
                    dst = os.path.join(face_dir[z], dfname)
                    crop_face(src, dst)
                    real_face_num += 1

    elif labels[z] == 1:

        for subdir, dirs, files in os.walk(frames_path[z]):
            for file in files:
                if file[-4:] == format_file[z]:
                    src = os.path.join(subdir, file)
                    dfname = file
                    dst = os.path.join(face_dir[z], dfname)
                    crop_face(src, dst)
                    fake_face_num += 1
