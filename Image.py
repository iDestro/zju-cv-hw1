import cv2


class Image:
    def __init__(self, filename):
        self.filename = filename
        self.label = self.filename.split('/')[-2]
        self.descriptors = self.__extract_descriptors()
        self.descriptors_size = len(self.descriptors)
        self.histogram = None
        self.tf_idf_weighted_histogram = None

    def __extract_descriptors(self):
        sift = cv2.xfeatures2d.SIFT_create()
        img = cv2.imread(self.filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(img, None)
        return descriptors

    def set_histogram(self, histogram):
        self.histogram = histogram

    def set_tf_idf_weighted_histogram(self, tf_idf_weighted_histogram):
        self.tf_idf_weighted_histogram = tf_idf_weighted_histogram
