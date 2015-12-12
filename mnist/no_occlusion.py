from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import numpy as np

batch_size = 128
nb_classes = 10
nb_epoch = 12

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 28, 28
# number of convolutional filters to use
nb_filters = 32
# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3

num_rows_obscured = 5

obscure = False

def obscure(image_matrix):
    obscured_matrix = np.copy(image_matrix)

    # Black out upper rectangle
    obscured_matrix[:, :num_rows_obscured, :] = 255
    return obscured_matrix

def show_image(image_vector):
    plt.imshow(image_vector.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

if obscure:
    X_train = obscure(X_train)
    X_test = obscure(X_test)


X_train = X_train.reshape(X_train.shape[0], 1, shapex, shapey)
X_test = X_test.reshape(X_test.shape[0], 1, shapex, shapey)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, 1, nb_conv, nb_conv, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
# the resulting image after conv and pooling is the original shape
# divided by the pooling with a number of filters for each "pixel"
# (the number of filters is determined by the last Conv2D)
model.add(Dense(nb_filters * (shapex / nb_pool) * (shapey / nb_pool), 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


