import cv2
import os
import shutil

Files_path = r"A"  # original whu dataset file path, change to B/label to process each item
labels_num = len(os.listdir(Files_path))
print(labels_num)

outpath = 'A_224'
if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)


def split(img, img_name):
    print(img_name)
    size = img.shape
    index = 0
    for i in range(size[0] // 224):
        for j in range(size[1] // 224):
            crop_img = img[i * 224:(i + 1) * 224, j * 224:(j + 1) * 224]
            cv2.imwrite(outpath + '/' + str(index) + '.png', crop_img)
            index = index + 1


for i in range(labels_num):
    image_dir = os.path.join(Files_path, str(os.listdir(Files_path)[i]))
    image_path = os.path.join(image_dir)
    img = cv2.imread(image_path)
    split(img, str(os.listdir(Files_path)[i]))
