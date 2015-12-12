import os
import gzip
import cPickle
import matplotlib.pyplot as plt 
import numpy as np

N = 2000
D = 784
K = 10

def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set

def show_image(image_vector):
    plt.imshow(image_vector.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

def show_vectors(vectors):
    for i in range(len(vectors)):
        show_image(vectors[i])

def obscure(image_vector):
    pixel_chart = image_vector.reshape(28, 28)
    pixel_chart[:10, :] = 1.0
    pixel_chart.reshape(-1, )
    return pixel_chart


X_data, Y_data = load_data('mnist.pkl.gz')
show_image(obscure(X_data[0]))


