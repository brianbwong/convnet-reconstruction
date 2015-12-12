from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import pickle

dataset = []

NUM_IMAGES = 1486
DIR_NAME = 'cats/'
# Assumed that the files 0.jpg, 1.jpg, ..., {NUM_IMAGES - 1}.jpg are located
# in DIR_NAME.

TRAIN_RATIO = 0.85
DEST_NAME = 'cats.dat'
shapex, shapey = 256, 256

def show_image(image_vector):
    im = image_vector.transpose(1,2,0)
    plt.imshow(im, interpolation='nearest')
    plt.show()

def crop(matrix):
    # input matrix has shape (x, y, 3)
    # with x >= shapex and y >= shapey
    x, y, _ = matrix.shape
    start_x = (x - shapex) / 2
    start_y = (y - shapey) / 2
    return matrix[start_x : start_x + shapex, start_y : start_y + shapey, :]

for counter in range(NUM_IMAGES):
    exists = True
    filename = DIR_NAME + str(counter) + '.jpg'
    try:
        im = Image.open(filename).convert('RGB')
        pix = im.load()
    except:
        exists = False

    if exists:
        # Check for generic image (when the actual image cannot be loaded)
        if im.size == (500, 374) and pix[250, 180] == (255, 255, 255):
            pass
        else:
            matrix = np.array(im)  # Now has shape (X, Y, 3)
            x, y, _ = matrix.shape
            if x >= shapex and y >= shapey:
                matrix = crop(matrix)
                matrix = matrix.transpose(2, 0, 1)
                dataset.append(matrix)

dataset = np.array(dataset)
num_images = dataset.shape[0]
train_thresh = int(TRAIN_RATIO * num_images)
training_set = dataset[:train_thresh]
test_set = dataset[train_thresh:]

print 'Creating dataset:'
print training_set.shape[0], 'training images'
print test_set.shape[0], 'test images'
with open(DEST_NAME, 'w') as f:
    pickle.dump([training_set, test_set], f)

