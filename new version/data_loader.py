import numpy as np
import struct
import scipy.ndimage.interpolation as distort
from itertools import chain
from array import array

class Loader:
    def __init__(self, source='./mnist'):
        self.source = source
        self.test_images_fname = 't10k-images.idx3-ubyte'
        self.test_labels_fname = 't10k-labels.idx1-ubyte'
        self.train_images_fname = 'train-images.idx3-ubyte'
        self.train_labels_fname = 'train-labels.idx1-ubyte'

    def load(self, images_path, labels_path):
        with open(images_path, 'rb') as f:
            magic_num, size, rows, columns = struct.unpack('>IIII', f.read(16))

            if magic_num != 2051:
                print('error')

            images = []
            for _ in range(size):
                image = np.fromstring(f.read(rows*columns), np.dtype('B'))\
                    .reshape((rows, columns))
                image = np.pad(image, ((2, 2), (2, 2)), 'constant')  # stretch
                images.append(image)

        with open(labels_path, 'rb') as f:
            magic_num, size = struct.unpack('>II', f.read(8))

            if magic_num != 2049:
                print('error')
            labels = array("B", f.read())

        return images, labels

    def load_test(self):
        return self.load(self.source + '/' + self.test_images_fname,
                         self.source + '/' + self.test_labels_fname)

    def load_train(self):
        return self.load(self.source + '/' + self.train_images_fname,
                         self.source + '/' + self.train_labels_fname)

    # with np.roll() it's eazy to shift in any direction
    # lost uint8
    # scipy nd image interpolation very slow
    # np.roll

    def animate(self, image):
        height, width = np.shape(image)
        animation_h = []
        animation_v = []

        img = image
        while img.any():
            animation_h.append(img)
            img = distort.shift(img, (0, -1))
        animation_h.reverse()

        img = image
        while img.any():
            img = distort.shift(img, (0, 1))
            animation_h.append(img)

        img = image
        while img.any():
            animation_v.append(img)
            img = distort.shift(img, (-1, 0))
        animation_v.reverse()

        img = image
        while img.any():
            img = distort.shift(img, (1, 0))
            animation_v.append(img)

        return animation_h, animation_v

    def simple_movie(self, images):
        movie = list(chain.from_iterable([list(chain.from_iterable(self.animate(img)))
                                          for img in images]))
        return movie

# test
if __name__ == "__main__":
    np.set_printoptions(threshold=2000, linewidth=300, precision=3)
    loader = Loader()
    images, labels = loader.load_train()
