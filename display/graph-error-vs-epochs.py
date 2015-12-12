import matplotlib.pyplot as plt  

DATASET = 'imagenet'
BASELINE_PRINTOUT = DATASET + '-icnn.txt'
ICNN_PRINTOUT = DATASET + '-baseline.txt'

CUTOFF = 75

with open(BASELINE_PRINTOUT, 'r') as f:
    data = f.read()

lines = data.split('\n')
connected_points = []
for line in lines:
    if not line.startswith('Epoch'):
        connected_points.append(float(line.split()[-1]))

with open(ICNN_PRINTOUT, 'r') as f:
    data = f.read()

lines = data.split('\n')
inverted_points = []
for line in lines:
    if not line.startswith('Epoch'):
        inverted_points.append(float(line.split()[-1]))

plt.plot(connected_points[:CUTOFF], label='Baseline Neural Net')
plt.plot(inverted_points[:CUTOFF], label='Inverted Conv Net')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction error')
plt.legend()
title = 'Baseline Neural Network vs. Inverted Convolutional Net on imagenet'
plt.title(title)

plt.show()