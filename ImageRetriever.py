import torch
import math


class ImageRetriever:
    def __init__(self, bag_of_visual_words):
        self.kmeans = bag_of_visual_words.kmeans
        self.images = bag_of_visual_words.images
        self.inverted_file_table = bag_of_visual_words.inverted_file_table
        self.generate_image_histogram = bag_of_visual_words.generate_image_histogram
        self.__generate_tf_idf_weighted_histogram()

        self.total_tf_idf_weighted_histogram = self.__generate_total_tf_idf_weighted_histogram()

    def __generate_tf_idf_weighted_histogram(self):
        for image in self.images:
            image.set_tf_idf_weighted_histogram(self.__tf_idf(image))

    def __tf_idf(self, image):
        k = self.kmeans.n_clusters
        tf_idf_weighted_histogram = torch.zeros([1, k])
        visual_words_num = torch.sum(image.histogram)
        for i in range(k):
            tf = image.histogram[:, i] / visual_words_num
            idf = math.log(len(self.images) / (len(self.inverted_file_table[i]) + 1))
            print('tf:', tf)
            print('idf', idf)
            tf_idf_weighted_histogram[:, i] = tf * idf
        return tf_idf_weighted_histogram

    def __generate_total_tf_idf_weighted_histogram(self):
        total_tf_idf_weighted_histogram = None
        for image in self.images:
            print(image.tf_idf_weighted_histogram.shape)
            if total_tf_idf_weighted_histogram is None:
                total_tf_idf_weighted_histogram = image.tf_idf_weighted_histogram
            else:
                total_tf_idf_weighted_histogram = torch.cat(
                    (total_tf_idf_weighted_histogram, image.tf_idf_weighted_histogram), dim=0)
        return total_tf_idf_weighted_histogram

    def retrieve(self, image):

        labels = self.kmeans.predict(torch.Tensor(image.descriptors))
        self.generate_image_histogram(image, labels)
        image.set_tf_idf_weighted_histogram(self.__tf_idf(image))
        m = self.total_tf_idf_weighted_histogram.shape[0]
        l = torch.zeros([m, 1])
        for i in range(m):
            l[i, :] = torch.sum((image.tf_idf_weighted_histogram-self.total_tf_idf_weighted_histogram[i])**2, dim=1)
        min_loss_image_index = torch.argmin(l, dim=0)
        return self.images[min_loss_image_index]
