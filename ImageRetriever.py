import torch


class ImageRetriever:
    def __init__(self, bag_of_visual_words):
        self.bag_of_visual_words = bag_of_visual_words
        self.kmeans = self.bag_of_visual_words.kmeans
        self.images = self.bag_of_visual_words.images
        self.inverted_file_table = self.bag_of_visual_words.inverted_file_table
        self.__generate_tf_idf_weighted_histogram()

        self.total_tf_idf_weighted_histogram = self.__generate_total_tf_idf_weighted_histogram()

    def __generate_tf_idf_weighted_histogram(self):
        for image in self.images:
            image.set_tf_idf_weighted_histogram(self.__tf_idf(image))

    def __tf_idf(self, image):
        k = self.kmeans.n_clusters
        tf_idf_weighted_histogram = torch.zeros(k)
        visual_words_num = torch.sum(image.histogram)
        for i in range(k):
            tf = image.histogram[i] / visual_words_num
            idf = len(self.images) / (len(self.inverted_file_table[i]) + 1)
            tf_idf_weighted_histogram[i] = tf * idf
        return tf_idf_weighted_histogram

    def __generate_total_tf_idf_weighted_histogram(self):
        total_tf_idf_weighted_histogram = None
        for image in self.images:
            if total_tf_idf_weighted_histogram is None:
                total_tf_idf_weighted_histogram = image.tf_idf_weighted_histogram
            else:
                total_tf_idf_weighted_histogram = torch.cat(
                    (total_tf_idf_weighted_histogram, image.tf_idf_weighted_histogram), dim=0)
        return total_tf_idf_weighted_histogram

    @staticmethod
    def loss(x, y):
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=0)
        l = (x - y) ** 2
        l = l.sum(dim=-1).squeeze()
        return l

    def retrieve(self, image):
        l = self.loss(image.tf_idf_weighted_histogram, self.total_tf_idf_weighted_histogram)
        min_loss_image_index = torch.argmin(l)
        return self.images[min_loss_image_index]
