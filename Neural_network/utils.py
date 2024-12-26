import numpy as np
import matplotlib.pyplot as plt
import torch

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    if z.ndim == 2:
        z = z.T
        z = z - np.max(z, axis=0, keepdims=True)
        s = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        return s.T
    C = np.max(z)
    return np.exp(z - C) / np.sum(np.exp(z - C))


def cross_entropy_error_batch(y_hat, y):
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(1, y_hat.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(y * np.log(y_hat + delta)) / batch_size


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_fashion_mnist_labels(labels):
    test_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]

def get_ROC_image(t, y_hat):
    if t.ndim != 1: t = t.argmax(axis=1)

    print(t)
    m_p = 0
    m_n = 0
    for i in range(len(t)):
        if t[i] == 5:
            m_p += 1
        else:
            m_n += 1

    y_s = y_hat
    t_s = t

    for i in range(len(t)):
        idx = i
        for j in range(len(t)):
            if y_s[idx][5] < y_s[j][5]:
                idx = j
        y_s[i], y_s[idx] = y_s[idx], y_s[i]
        t_s[i], t_s[idx] = t_s[idx], t_s[i]

    x_image = [0.0]
    y_image = [0.0]

    for i in range(len(t)):
        if t_s[i] == 5:
            x_image.append(x_image[-1])
            y_image.append(y_image[-1] + 1 / m_p)
        else:
            x_image.append(x_image[-1] + 1 / m_n)
            y_image.append(y_image[-1])

    print(x_image[0:100])
    print(y_image[0:100])

    plt.plot(x_image, y_image)
    plt.show()

if __name__ == '__main__':
    input_data = np.arange(48).reshape(2, 2, 3, 4)
    print(input_data)
    col = im2col(input_data, 2, 2)
    print(col)