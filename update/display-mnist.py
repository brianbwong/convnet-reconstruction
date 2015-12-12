import pickle
import numpy as np
import matplotlib.pyplot as plt 

DATAFILE = 'connected-mnist-preds.dat'
obscure_start_row = 13
obscure_end_row = 18

def show_image(image_vector):
    plt.imshow(image_vector.reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

original_images, predicted_images = pickle.load(open(DATAFILE, 'rb'))

for i in [13, 15, 16, 22, 24]:
    show_image(original_images[i])
    show_image(predicted_images[i])