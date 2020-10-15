
import argparse

parser = argparse.ArgumentParser(description='Image retrieve base on bag of visual words')
parser.add_argument('--image_path', '-t', default=None, type=str, help='choose a image to retrieve')
parser.add_argument('--image_database_path', '-d', default="./dataset", type=str, help='choose a image database(just a path)')
args = parser.parse_args()


if __name__ == '__main__':
    pass