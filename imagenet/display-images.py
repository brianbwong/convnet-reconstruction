import pickle
import numpy as np
import matplotlib.pyplot as plt 

DATAFILE = 'cat-preds-6layers.dat'
obscure_start_row = 100
obscure_end_row = 116

shapex, shapey = 256, 256

def display(original, pred):
    original_shaped = original.reshape(3, shapex, shapey)
    pred_shaped = pred.reshape(3, shapex, shapey)
    reconstructed = np.concatenate((original_shaped[:, :obscure_start_row, :], pred_shaped[:, obscure_start_row:obscure_end_row, :]), axis=1)
    reconstructed = np.concatenate((reconstructed, original_shaped[:, obscure_end_row:, :]), axis=1)
    show_image(original_shaped)
    show_image(reconstructed) 

def show_image(image_vector):
    im = image_vector.reshape(3, shapex, shapey)
    im = im.transpose(1,2,0)
    plt.imshow(im, interpolation='nearest')
    plt.show()

original_images, predicted_images = pickle.load(open(DATAFILE, 'rb'))

for i in range(0, 1):
    display(original_images[i], predicted_images[i])
    show_image(predicted_images[i])