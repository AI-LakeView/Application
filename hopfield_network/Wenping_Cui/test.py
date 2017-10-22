import numpy as np
import matplotlib.pyplot as plt

import load_MNIST_data

# load data
training_data, validation_data, test_data = load_MNIST_data.load_data_wrapper()
training_data = list(training_data)

# take a look one data sample
data_sample = training_data[0]

# image part
data_img = data_sample[0]
plt.figure(1)
plt.imshow(np.reshape(data_img, [28, 28]), cmap='Greys')
plt.show()

# label part
data_label = data_sample[1]
plt.figure(2)
plt.imshow(data_label, cmap = 'Greys')
plt.yticks(np.arange(10))
plt.xticks(np.arange(0))
plt.show()

# concatenate these two parts
data_full = np.concatenate((data_img, data_label), axis=0)

# binary img
data_img_b = data_img
data_img_b[data_img_b < 0.5] = 0
data_img_b[data_img_b > 0.5] = 1
plt.figure(3)
plt.imshow(np.reshape(data_img_b, [28, 28]), cmap='Greys')
plt.show()