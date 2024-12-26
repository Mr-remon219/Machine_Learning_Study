import numpy as np
from data_utils import load_mnist
from network import *
from utils import *
from tqdm import *
import pickle
from optimizer import *
from trainer import *
import matplotlib.pyplot as plt


def test(X_test, y_test, idx):
    with open("params.pkl", "rb") as f:
        params = pickle.load(f)
        net = SimpleConvNet(input_dim=(1, 28, 28), hidden_size=50, output_size=10, params=params)
        y_pred_labels = np.argmax(net.predict(X_test), axis=1)
        print(net.accuracy(X_test, y_test))
        print("预测数字为:", y_pred_labels[idx])
        y_test_labels = np.argmax(y_test, axis=1)
        print("真实数字为：", y_test_labels[idx])


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)
    with open("params.pkl", "rb") as f:
        params = pickle.load(f)
        net = SimpleConvNet(input_dim=(1, 28, 28), hidden_size=50, output_size=10, params=params)
        get_ROC_image(t_test, net.predict(x_test))

