import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


sift = cv2.xfeatures2d.SIFT_create()
images_path = './dataset/boat'
list_dir = os.listdir(images_path)
descriptors = None
for i in list_dir:
    if 'pgm' in i:
        cur_img_path = images_path+os.sep+i
        print(cur_img_path)
        print(os.path.exists(cur_img_path))
        img = cv2.imread(cur_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sub_descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            descriptors = sub_descriptors
        else:
            descriptors = np.concatenate((descriptors, sub_descriptors), axis=0)

np.save('descriptors', descriptors)
