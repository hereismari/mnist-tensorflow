import os
from six.moves.urllib.request import urlretrieve

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

# change the path!!!
WORK_DIRECTORY = "/home/marianne-linhares/DeepLearning/TensorFlow/mnist/data/"

def download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)

    filepath = os.path.join(WORK_DIRECTORY, filename)

    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)

    return filepath


if __name__ == '__main__':
    train_data_filename =   download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename =    download('t10k-images-idx3-ubyte.gz')
    test_labels_filename =  download('t10k-labels-idx1-ubyte.gz')

