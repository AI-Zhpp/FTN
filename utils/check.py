from PIL import Image
import os

Files_path = r"../try"
labels_num = len(os.listdir(Files_path))
print(labels_num)
import numpy as np

# check if the masks are correct
for i in range(labels_num):
    image_dir = os.path.join(Files_path, str(os.listdir(Files_path)[i]))
    image_path = os.path.join(image_dir)
    img = Image.open(image_path)
    print(np.array(img).shape)
    print(np.array(img).shape)
    # if the mask is abnormal, the item will be printed
    if np.array(img).sum() % 5 != 0:
        print(np.array(img).sum())
