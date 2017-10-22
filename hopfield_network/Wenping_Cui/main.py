import numpy as np
import matplotlib.pyplot as plt

import load_MNIST_data
from hf_net import HopfNet

# load data
training_data, validation_data, test_data = load_MNIST_data.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# initiate a net
h_net = HopfNet(784)

# form memories
memory_img_count = 60
for i in range(memory_img_count):
    print(i)
    input_sample = training_data[i]
    input_img = input_sample[0]

    input_img[input_img < 0.5] = 0
    input_img[input_img > 0.5] = 1

    h_net.train(input_img)

# retrieve memory
test_in_sample = test_data[1]
test_in_img = test_in_sample[0]

test_in_img[test_in_img < 0.5] = 0
test_in_img[test_in_img > 0.5] = 1

test_out_img = h_net.retrieve(test_in_img)

# show the img
plt.figure(1)
plt.imshow(np.reshape(test_in_img, [28, 28]), cmap='Greys')
plt.show()

#
plt.figure(2)
plt.imshow(np.reshape(test_out_img, [28, 28]), cmap='Greys')
plt.show()