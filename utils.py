import os
from Image import Image
import cv2
import matplotlib.pyplot as plt


def load_images(path):
    images = []
    dirs = os.listdir(path)
    for cur_dir in dirs:
        filenames = os.listdir(path + '/' + cur_dir)
        for filename in filenames:
            images.append(Image(path + '/' + cur_dir + '/' + filename))
    return images


def load_image(path):
    return Image(path)


def imshow(image):
    img = cv2.imread(image.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


