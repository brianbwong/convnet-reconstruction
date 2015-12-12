from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import pickle
sys.path.append('/home/brianwong/keras/')
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import numpy as np

batch_size = 128
nb_classes = 10
nb_epoch = 200

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 28, 28

# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3

num_train = 60000

obscure_start_row = 13
obscure_end_row = 18

def obscure(image_matrix):
    obscured_matrix = np.copy(image_matrix)

    # Black out rectangle
    obscured_matrix[:, obscure_start_row:obscure_end_row, :] = 255
    return obscured_matrix

def show_image(image_vector):
    plt.imshow(image_vector.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

(X_train_original, _), (X_test_original, _) = mnist.load_data()

X_train_original = X_train_original[:num_train]

X_train_obscured = obscure(X_train_original)
X_test_obscured = obscure(X_test_original)


#for i in range(10):
#    show_image(X_train[i])

X_train_original = X_train_original.reshape(X_train_original.shape[0], -1)
X_test_original = X_test_original.reshape(X_test_original.shape[0], -1)
X_train_original = X_train_original.astype("float32")
X_test_original = X_test_original.astype("float32")
X_train_original /= 255
X_test_original /= 255

X_train_obscured = X_train_obscured.reshape(X_train_obscured.shape[0], 1, shapex, shapey)
X_test_obscured = X_test_obscured.reshape(X_test_obscured.shape[0], 1, shapex, shapey)
X_train_obscured = X_train_obscured.astype("float32")
X_test_obscured = X_test_obscured.astype("float32")
X_train_obscured /= 255
X_test_obscured /= 255

print(X_train_obscured.shape[0], 'train samples')
print(X_test_obscured.shape[0], 'test samples')

model = Sequential()

model.add(Convolution2D(64, 1, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 64, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 128, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(1, 64, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('sigmoid'))
model.add(Flatten())

model.compile(loss='mean_squared_error', optimizer='adadelta')

model.fit(
    X_train_obscured, 
    X_train_original, 
    batch_size=batch_size, 
    nb_epoch=nb_epoch, 
    show_accuracy=False, 
    verbose=1, 
    validation_data=(X_test_obscured, X_test_original)
)

mini_test = X_test_obscured[:128]
preds = model.predict(mini_test)

data = [mini_test, preds]
#pickle.dump(data, open('mnist-preds.dat', 'wb'))


'''
score = model.evaluate(X_test_obscured, X_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

