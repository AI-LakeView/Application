import argparse
from Hopfield import mnist_gen, Hopfield


if __name__ == '__main__':
    '''
    Example:
        python main.py mnist_train.csv mnist_test.csv 255
        python main.py simplified_digit.csv simplified_digit.csv 1
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file_name')
    parser.add_argument('test_file_name')
    parser.add_argument('im_max')

    args = parser.parse_args()
    train_file_name = args.test_file_name
    test_file_name = args.test_file_name
    im_max = float(args.im_max)

    h = Hopfield()
    h.train_on_file(mnist_gen(train_file_name, im_max), train_max=100)
    h.test_on_file(mnist_gen(test_file_name, im_max))

    print('done')