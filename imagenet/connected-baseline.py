from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import pickle
import sys
sys.path.append('/home/brianwong/keras/')
from keras.datasets import cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import numpy as np

data_file = 'cats.dat'

batch_size = 1
nb_epoch = 75

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 256, 256

# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3

obscure_start_row = 100
obscure_end_row = 116

#l2_penalty = 0.01

def obscure(image_matrix):

    obscured_matrix = np.copy(image_matrix)

    # Black out rectangle
    obscured_matrix[:, :, obscure_start_row:obscure_end_row, :] = 0
    return obscured_matrix

def show_image(image_vector):
    im = image_vector.reshape(3, shapex, shapey)
    im = im.transpose(1,2,0)
    plt.imshow(im, interpolation='nearest')
    plt.show()

with open(data_file) as f:
    X_train_original, X_test_original = pickle.load(f)
    X_train_original = X_train_original[:]
    X_test_original = X_test_original[:]

X_train_obscured = obscure(X_train_original)
X_test_obscured = obscure(X_test_original)

X_train_original = X_train_original.reshape(X_train_original.shape[0], -1)
X_test_original = X_test_original.reshape(X_test_original.shape[0], -1)
X_train_original = X_train_original.astype("float32")
X_test_original = X_test_original.astype("float32")
X_train_original /= 255
X_test_original /= 255

X_train_obscured = X_train_obscured.reshape(X_train_obscured.shape[0], -1)
X_test_obscured = X_test_obscured.reshape(X_test_obscured.shape[0], -1)
X_train_obscured = X_train_obscured.astype("float32")
X_test_obscured = X_test_obscured.astype("float32")
X_train_obscured /= 255
X_test_obscured /= 255

print('Training set shape:', X_train_obscured.shape)
print('Test set shape:', X_test_obscured.shape)


model = Sequential()

model.add(Dense(shapex * shapey * 3, 8))
model.add(Activation('relu'))
model.add(Dense(8, shapex * shapey * 3))
model.add(Activation('sigmoid'))
model.add(Flatten())


ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

model.compile(loss='mean_squared_error', optimizer=ada)

model.fit(
    X_train_obscured, 
    X_train_original, 
    batch_size=batch_size, 
    nb_epoch=nb_epoch, 
    show_accuracy=False, 
    verbose=1, 
    validation_data=(X_test_obscured, X_test_original)
)

json_string = model.to_json()
open('cat-model-4layers-75epochs-baseline.json', 'w').write(json_string)
model.save_weights('cat-weights-4layers-75epochs-baseline.h5')

mini_test = X_test_obscured[:10]
preds = model.predict(mini_test)

data = [mini_test, preds]
pickle.dump(data, open('cat-preds-4layers-75epochs-baseline.dat', 'wb'))
