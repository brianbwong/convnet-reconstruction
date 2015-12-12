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
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import numpy as np

data_file = 'cats.dat'

batch_size = 32
nb_epoch = 600

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 256, 256

# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 8

obscure_start_row = 100
obscure_end_row = 130

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

X_train_original = X_train_original
X_train_obscured = obscure(X_train_original)
X_test_obscured = obscure(X_test_original)

X_train_original = X_train_original.reshape(X_train_original.shape[0], -1)
X_test_original = X_test_original.reshape(X_test_original.shape[0], -1)
X_train_original = X_train_original.astype("float32")
X_test_original = X_test_original.astype("float32")
X_train_original /= 255
X_test_original /= 255

X_train_obscured = X_train_obscured.reshape(X_train_obscured.shape[0], 3, shapex, shapey)
X_test_obscured = X_test_obscured.reshape(X_test_obscured.shape[0], 3, shapex, shapey)
X_train_obscured = X_train_obscured.astype("float32")
X_test_obscured = X_test_obscured.astype("float32")
X_train_obscured /= 255
X_test_obscured /= 255

print('Training set shape:', X_train_obscured.shape)
print('Test set shape:', X_test_obscured.shape)

model = Graph()

model.add_input(name='input', ndim=4)

model.add_node(Convolution2D(64, 3, 3, 3, border_mode='same'), name='conv1_1', input='input')
model.add_node(Activation('relu'), name='relu1_1', input='conv1_1')
model.add_node(Convolution2D(64, 64, 3, 3, border_mode='same'), name='conv1_2', input='relu1_1')
model.add_node(Activation('relu'), name='relu1_2', input='conv1_2')
model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool1', input='relu1_2')

model.add_node(Convolution2D(128, 64, 3, 3, border_mode='same'), name='conv2_1', input='pool1')
model.add_node(Activation('relu'), name='relu2_1', input='conv2_1')
model.add_node(Convolution2D(128, 128, 3, 3, border_mode='same'), name='conv2_2', input='relu2_1')
model.add_node(Activation('relu'), name='relu2_2', input='conv2_2')
model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool2', input='relu2_2')

model.add_node(Convolution2D(256, 128, 3, 3, border_mode='same'), name='conv3_1', input='pool2')
model.add_node(Activation('relu'), name='relu3_1', input='conv3_1')
model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='conv3_2', input='relu3_1')
model.add_node(Activation('relu'), name='relu3_2', input='conv3_2')
model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='conv3_3', input='relu3_2')
model.add_node(Activation('relu'), name='relu3_3', input='conv3_3')
model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool3', input='relu3_3')

model.add_node(Convolution2D(512, 256, 3, 3, border_mode='same'), name='conv4_1', input='pool3')
model.add_node(Activation('relu'), name='relu4_1', input='conv4_1')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv4_2', input='relu4_1')
model.add_node(Activation('relu'), name='relu4_2', input='conv4_2')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv4_3', input='relu4_2')
model.add_node(Activation('relu'), name='relu4_3', input='conv4_3')
model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool4', input='relu4_3')

model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_1', input='pool4')
model.add_node(Activation('relu'), name='relu5_1', input='conv5_1')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_2', input='relu5_1')
model.add_node(Activation('relu'), name='relu5_2', input='conv5_2')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_3', input='relu5_2')
model.add_node(Activation('relu'), name='relu5_3', input='conv5_3')
model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool5', input='relu5_3')

#############

model.add_node(UpSample2D(size=(2, 2)), name='xpool5', input='pool5')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='xconv5_1', input='xpool5')
model.add_node(Activation('relu'), name='xrelu5_1', input='xconv5_1')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='xconv5_2', input='xrelu5_1')
model.add_node(Activation('relu'), name='xrelu5_2', input='xconv5_2')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='xconv5_3', input='xrelu5_2')
model.add_node(Activation('relu'), name='xrelu5_3', input='xconv5_3')

model.add_node(UpSample2D(size=(2, 2)), name='xpool4', input='xrelu5_3')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='xconv4_1', input='xpool4')
model.add_node(Activation('relu'), name='xrelu4_1', input='xconv4_1')
model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='xconv4_2', input='xrelu4_1')
model.add_node(Activation('relu'), name='xrelu4_2', input='xconv4_2')
model.add_node(Convolution2D(256, 512, 3, 3, border_mode='same'), name='xconv4_3', input='relu4_2')
model.add_node(Activation('relu'), name='xrelu4_3', input='xconv4_3')

model.add_node(UpSample2D(size=(2, 2)), name='xpool3', input='xrelu4_3')
model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='xconv3_1', input='xpool3')
model.add_node(Activation('relu'), name='xrelu3_1', input='xconv3_1')
model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='xconv3_2', input='xrelu3_1')
model.add_node(Activation('relu'), name='xrelu3_2', input='xconv3_2')
model.add_node(Convolution2D(128, 256, 3, 3, border_mode='same'), name='xconv3_3', input='xrelu3_2')
model.add_node(Activation('relu'), name='xrelu3_3', input='xconv3_3')

model.add_node(UpSample2D(size=(2, 2)), name='xpool2', input='xrelu3_3')
model.add_node(Convolution2D(128, 128, 3, 3, border_mode='same'), name='xconv2_1', input='xpool2')
model.add_node(Activation('relu'), name='xrelu2_1', input='xconv2_1')
model.add_node(Convolution2D(64, 128, 3, 3, border_mode='same'), name='xconv2_2', input='xrelu2_1')
model.add_node(Activation('relu'), name='xrelu2_2', input='xconv2_2')

model.add_node(UpSample2D(size=(2, 2)), name='xpool1', input='relu2_2')
model.add_node(Convolution2D(64, 64, 3, 3, border_mode='same'), name='xconv1_1', input='xpool1')
model.add_node(Activation('relu'), name='xrelu1_1', input='xconv1_1')
model.add_node(Convolution2D(3, 64, 3, 3, border_mode='same'), name='xconv1_2', input='xrelu1_1')
model.add_node(Activation('relu'), name='xrelu1_2', input='xconv1_2')

model.add_output(name='output', input='xrelu1_2')

model.compile(loss={'output': 'mse'}, optimizer='adadelta')

#model.compile(loss='mean_squared_error', optimizer='adadelta')
model.fit({'input': X_train_obscured, 'output': X_train_original}, batch_size=batch_size)
'''
model.fit(
    X_train_obscured, 
    X_train_original, 
    batch_size=batch_size, 
    nb_epoch=nb_epoch, 
    show_accuracy=False, 
    verbose=1, 
    validation_data=(X_test_obscured, X_test_original)
)
'''
mini_test = X_test_obscured[:16]
preds = model.predict(mini_test)

data = [mini_test, preds]
pickle.dump(data, open('cat-preds.dat', 'wb'))
