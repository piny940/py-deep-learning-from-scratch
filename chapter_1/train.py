# coding: utf-8
import sys
sys.path.append('/Users/ansai/Documents/VScode/deep-learning/deep-learning-from-scratch')  # 親ディレクトリのファイルをインポートするための設定
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from nnw import TwoLayerNet
import numpy as np
from matplotlib import pyplot as plt


# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)

f0 = np.linspace(-1, 1, 200)
f1 = np.linspace(-1, 1, 200)
f0, f1 = np.meshgrid(f0, f1)
data = np.hstack([f0.reshape(-1, 1), f1.reshape(-1, 1)])
pred = model.predict(data)
col = np.zeros_like(f0)
print(pred)
for i in range(col.shape[0]):
    for j in range(col.shape[1]):
        col[i][j] = pred[i * col.shape[1] + j].argmax()
plt.contourf(f0, f1, col)

# データ点のプロット
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])

plt.show()
