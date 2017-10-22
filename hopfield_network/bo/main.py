import re
import sys
import logging
import numpy as np
import argparse


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mnist_gen(filename, thre=0.5):
    with open(filename,'r') as f:
        for line in f:
            output = line.strip().split(',')
            
            label = int(output[0])
            data = [1 if int(x)/255. > thre else 0 for x in output[1:]]

            label_one_hot = [0 for _ in range(10)]
            label_one_hot[label] = 1

            yield label, label_one_hot, data


class Hopfield():

    def __init__(self):
        self.dim = None
        self.T = None

    def train(self, filename, train_max=1000):
        train_gen = mnist_gen(filename)
        for i_data, (label, label_one_hot, data) in enumerate(train_gen):
            if not self.dim:
                self.setup(data)

            if i_data % 10 == 0:
                logging.info(i_data)

            self._train(label, label_one_hot, data)

            if i_data > train_max:
                break

    def _train(self, label, label_one_hot, data):
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

    def test(self, filename, test_max=10):
        logging.info('starting test')
        
        test_gen = mnist_gen(filename)
        for i_data, (label, label_one_hot, data) in enumerate(test_gen):
            logging.info(i_data)
            self._test(data)

    def _test(self, data, runs=None):
        if not runs:
            runs = 100 * self.dim

        V = data.copy()
        for pos in np.random.randint(self.dim, size=runs):
            if np.sum(self.T[pos] * V) > 0:
                V[pos] = 1
            else:
                V[pos] = 0

        import pdb; pdb.set_trace()
        return V


if __name__ == '__main__':
    '''
    Example:
        python main.py mnist_train.csv mnist_test.csv
        python main.py simplified_digit.csv simplified_digit.csv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file_name')
    parser.add_argument('test_file_name')

    args = parser.parse_args()
    train_file_name = args.test_file_name
    test_file_name = args.test_file_name

    h = Hopfield()

    h.train(train_file_name, train_max=100)

    h.test(test_file_name)

    print('done')