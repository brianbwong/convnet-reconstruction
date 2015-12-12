import pickle
import numpy as np 
import matplotlib.pyplot as plt 

TARGET_FILE = 'cats.dat'

def show_image(image_vector):
    im = image_vector.transpose(1,2,0)
    plt.imshow(im, interpolation='nearest')
    plt.show()

with open(TARGET_FILE) as f:
    training_set, test_set = pickle.load(f)
    print training_set.shape
    show_image(training_set[40])