import pickle
import numpy as np
import matplotlib.pyplot as plt 

DATAFILE = 'preds.dat'
obscure_start_row = 13
obscure_end_row = 18

def display(original, pred):
    original_shaped = original.reshape(3, 32, 32)
    pred_shaped = pred.reshape(3, 32, 32)
    reconstructed = np.concatenate((original_shaped[:, :obscure_start_row, :], pred_shaped[:, obscure_start_row:obscure_end_row, :]), axis=1)
    reconstructed = np.concatenate((reconstructed, original_shaped[:, obscure_end_row:, :]), axis=1)
    show_image(original_shaped)
    show_image(reconstructed) 

def show_image(image_vector):
    im = image_vector.reshape(3, 32, 32)
    im = im.transpose(1,2,0)
    plt.imshow(im, interpolation='nearest')
    plt.show()

original_images, predicted_images = pickle.load(open(DATAFILE, 'rb'))

for i in [10, 13, 16]:
    display(original_images[i], predicted_images[i])