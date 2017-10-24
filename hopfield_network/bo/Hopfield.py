import re
import sys
import logging
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mnist_gen(filename, thre=0.5, im_max=255.):
    with open(filename,'r') as f:
        for line in f:
            output = line.strip().split(',')
            
            label = int(output[0])
            data = [1 if int(x)/im_max > thre else 0 for x in output[1:]]

            label_one_hot = [0 for _ in range(10)]
            label_one_hot[label] = 1

            yield label, label_one_hot, data


class Hopfield():

    def __init__(self):
        self.dim = None
        self.T = None

    def train_on_file(self, train_data_gen, train_max=1000):
        for i_data, (label, label_one_hot, data) in enumerate(train_data_gen):
            if not self.dim:
                self.setup(data)

            if i_data % 10 == 0:
                logging.info(i_data)

            self.train(label, label_one_hot, data)

            if i_data > train_max:
                break

    def train(self, label, label_one_hot, data):
        data = np.array(data)
        for i in range(self.dim):
            if data[i] > 0:
                self.T[i] += (2 * data - 1)
            else:
                self.T[i] -= (2 * data - 1)
            self.T[(i, i)] = 0
        return

    def setup(self, data):
        self.dim = len(data)
        self.T = np.zeros((self.dim, self.dim))
        return

    def test_on_file(self, test_data_gen, test_max=10):
        logging.info('starting test')
        
        for i_data, (label, label_one_hot, data) in enumerate(test_data_gen):
            logging.info(i_data)
            self.test(data)

    def test(self, data, runs=None):
        if not runs:
            runs = 100 * self.dim

        V = data.copy()
        for pos in np.random.randint(self.dim, size=runs):
            if np.sum(self.T[pos] * V) > 0:
                V[pos] = 1
            else:
                V[pos] = 0

        return V