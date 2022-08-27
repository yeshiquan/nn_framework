import numpy as np

from layers import *
from losses import CrossEntropyLoss
from activations import ReLU, Softmax
from net import Net

n_class_size = 100
r = 2
X1_offset = np.random.rand(n_class_size, 2) - 0.5
np.sqrt(np.sum(X1_offset ** 2, axis=1, keepdims=True))
X1_offset = r * X1_offset / np.sqrt(np.sum(X1_offset ** 2, axis=1, keepdims=True))
X1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size) + X1_offset
X2 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size)

X = np.concatenate((X1, X2))
Y_labels = np.array([0] * n_class_size + [1] * n_class_size)

net = Net(
    layers=[Linear(2, 4), ReLU(), Linear(4, 2), Softmax()], loss=CrossEntropyLoss()
)

print(X)
print(Y_labels)
print(net)

n_epochs = 10000
for epoch_idx in range(n_epochs):
    # 开始前向传播
    out = net(X)  # out=(200,2)
    print(out)
    # prediction accuracy
    pred = np.argmax(out, axis=1)  # pred=(200)
    loss = net.loss(out, Y_labels) # Y_labels=(200)

    if epoch_idx % 200 == 0:
        print("Epoch: %d" % epoch_idx, end =" ")
        print("accuracy: %1.4f" % (1 - np.abs(pred - Y_labels).sum() / 200), end =" ")
        print("loss: %1.4f" % loss)

    # 反向传播，计算各个参数给最终误差的影响力(梯度)
    grad = net.backward()

    # 根据各个参数的梯度+学习率更新参数，优化器是简单的随机梯度下降
    net.update_weights(0.1)