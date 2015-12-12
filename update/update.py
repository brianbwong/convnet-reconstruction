import matplotlib.pyplot as plt  

with open('connected.txt', 'r') as f:
    data = f.read()

lines = data.split('\n')
connected_points = []
for line in lines:
    if not line.startswith('Epoch'):
        connected_points.append(float(line.split()[-1]))

with open('inverted.txt', 'r') as f:
    data = f.read()

lines = data.split('\n')
inverted_points = []
for line in lines:
    if not line.startswith('Epoch'):
        inverted_points.append(float(line.split()[-1]))

plt.plot(connected_points[:200], label='Baseline Neural Net')
plt.plot(inverted_points, label='Inverted Conv Net')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction error')
plt.legend()
plt.title('Baseline Neural Network vs. Inverted Convolutional Net')

plt.show()