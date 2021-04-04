import dwt_path
import os


base_dir = dwt_path.base_dir_WT

train_dir = dwt_path.train_dir_WT
validation_dir = dwt_path.validation_dir_WT
test_dir = dwt_path.test_dir_WT

train_real_dir = dwt_path.train_real_dir
train_fake_dir = dwt_path.train_fake_dir
validation_real_dir = dwt_path.validation_real_dir
validation_fake_dir = dwt_path.validation_fake_dir
test_real_dir = dwt_path.test_real_dir
test_fake_dir = dwt_path.test_fake_dir

train_real_videos = []
validation_real_videos = []
test_real_videos = []
train_fake_videos = []
validation_fake_videos = []
test_fake_videos = []


def video_record():
    for f in ['cA/', 'cH/', 'cV/', 'cD/']:
        for subdir, dirs, files in os.walk(train_real_dir[f]):
            for file in files:
                if file.split('%')[0] not in train_real_videos:
                    train_real_videos.append(file.split('%')[0])

        for subdir, dirs, files in os.walk(validation_real_dir[f]):
            for file in files:
                if file.split('%')[0] not in validation_real_videos:
                    validation_real_videos.append(file.split('%')[0])

        for subdir, dirs, files in os.walk(test_real_dir[f]):
            for file in files:
                if file.split('%')[0] not in test_real_videos:
                    test_real_videos.append(file.split('%')[0])

        for subdir, dirs, files in os.walk(train_fake_dir[f]):
            for file in files:
                if file.split('%')[0] not in train_fake_videos:
                    train_fake_videos.append(file.split('%')[0])

        for subdir, dirs, files in os.walk(validation_fake_dir[f]):
            for file in files:
                if file.split('%')[0] not in validation_fake_videos:
                    validation_fake_videos.append(file.split('%')[0])

        for subdir, dirs, files in os.walk(test_fake_dir[f]):
            for file in files:
                if file.split('%')[0] not in test_fake_videos:
                    test_fake_videos.append(file.split('%')[0])

    print(train_real_videos)
    print(validation_real_videos)
    print(test_real_videos)
    print(train_fake_videos)
    print(validation_fake_videos)
    print(test_fake_videos)


video_record()

for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    for subdir, dirs, files in os.walk(validation_real_dir[f]):
        for file in files:
            if file.split('%')[0] in train_real_videos:
                os.remove(subdir+'/'+file)

    for subdir, dirs, files in os.walk(test_real_dir[f]):
        for file in files:
            if file.split('%')[0] in validation_real_videos:
                os.remove(subdir+'/'+file)

    for subdir, dirs, files in os.walk(validation_fake_dir[f]):
        for file in files:
            if file.split('%')[0] in train_fake_videos:
                os.remove(subdir+'/'+file)

    for subdir, dirs, files in os.walk(test_fake_dir[f]):
        for file in files:
            if file.split('%')[0] in validation_fake_videos:
                os.remove(subdir+'/'+file)

train_real_videos = []
validation_real_videos = []
test_real_videos = []
train_fake_videos = []
validation_fake_videos = []
test_fake_videos = []
video_record()


for f in ['cA/', 'cH/', 'cV/', 'cD/']:
    print('total training real images:', len(os.listdir(train_real_dir[f])))
    print('total training fake images:', len(os.listdir(train_fake_dir[f])))
    print('total validation real images:', len(os.listdir(validation_real_dir[f])))
    print('total validation fake images:', len(os.listdir(validation_fake_dir[f])))
    print('total test real images:', len(os.listdir(test_real_dir[f])))
    print('total test fake images:', len(os.listdir(test_fake_dir[f])))

