import cv2
import torch
import matplotlib.pyplot as plt

# read images
img1 = cv2.imread('./dataset/boat/img1.pgm')
img2 = cv2.imread('./dataset/boat/img2.pgm')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# sift
sift = cv2.xfeatures2d.SIFT_create()
key_points_1, descriptors_1 = sift.detectAndCompute(img1, None)
key_points_2, descriptors_2 = sift.detectAndCompute(img2, None)
# feature matching
print(descriptors_1.shape)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, key_points_1, img2, key_points_2, matches[:50], img2, flags=2)
plt.imshow(img3)
plt.show()
