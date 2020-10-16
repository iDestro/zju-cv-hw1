import torch
from Kmeans import KMeans
from BagOfVisualWords import BagOfVisualWords
from ImageRetriever import ImageRetriever
from utils import load_images, load_image, imshow
import argparse

parser = argparse.ArgumentParser(description='Image retrieve base on bag of visual words')
parser.add_argument('--name', '-n', type=str, help='choose a image from ./dataset/test/* to retrieve')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    kmeans = KMeans(n_clusters=70, device=device)
    image_path = './dataset/test/'+args.name
    target_image = load_image(image_path)
    images = load_images('./dataset/train')
    image_retriever = ImageRetriever(BagOfVisualWords(images=images, kmeans=kmeans))
    result = image_retriever.retrieve(target_image)
    imshow(result)
