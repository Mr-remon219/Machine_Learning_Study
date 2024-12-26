import numpy as np
import os
import gzip
import pickle



dataset_dir = r'C:\Users\hp\PycharmProjects\pythonProject\MNIST_dataset'

filenames = {
    'train_imgs': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_imgs': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}
saved_file = 'mnist.pkl'

def init__dir(dir_name):
    dataset_dir = dir_name

def load_img(file_name):
    file_path = dataset_dir + '/' + file_name
    print(f'将{file_path}转成numpy数组...')
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)

    data = data.reshape(-1, 784)
    return data

def load_label(file_name):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data


def convert_2_numpy():
    dataset = {}
    dataset['train_imgs'] = load_img(filenames['train_imgs'])
    dataset['train_labels'] = load_label(filenames['train_labels'])
    dataset['test_imgs'] = load_img(filenames['test_imgs'])
    dataset['test_labels'] = load_label(filenames['test_labels'])
    return dataset


def init_mnist():
    dataset = convert_2_numpy()
    print('将数据集转换成pickle文件...')
    with open(dataset_dir + '/' + saved_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print(f'{saved_file}')

def _change_one_hot_label(x):
    T = np.zeros((x.size, 10))
    for idx, row in enumerate(T):
        row[x[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/' + saved_file):
        init_mnist()

    with open(dataset_dir + '/' + saved_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_imgs', 'test_imgs'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset['train_labels'] = _change_one_hot_label(dataset['train_labels'])
        dataset['test_labels'] = _change_one_hot_label(dataset['test_labels'])

    if not flatten:
        for key in ('train_imgs', 'test_imgs'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)


    return (dataset['train_imgs'], dataset['train_labels']), (dataset['test_imgs'], dataset['test_labels'])

