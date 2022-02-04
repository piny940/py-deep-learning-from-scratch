import sys
sys.path.append('/Users/ansai/Documents/VScode/deep-learning')
import numpy as np
from dataset import spiral
from chapter_1.nnw import TwoLayerNet
from common.optimizer import SGD
import time
from matplotlib import pyplot as plt

max_epoch=300
batch_size=30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
max_grad=None
eval_interval=10
data_size = len(x)
max_iters = data_size // batch_size
model, optimizer = TwoLayerNet(2,hidden_size,3), SGD(learning_rate)
total_loss = 0
loss_count = 0
loss_list = []
current_epoch = 0

start_time = time.time()
for epoch in range(max_epoch):
# シャッフル
    idx = np.random.permutation(np.arange(data_size))
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

        # 評価
        if (eval_interval is not None) and (iters % eval_interval) == 0:
            avg_loss = total_loss / loss_count
            elapsed_time = time.time() - start_time
            print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                    % (current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
            loss_list.append(float(avg_loss))
            total_loss, loss_count = 0, 0

    current_epoch += 1
    
x = np.arange(len(loss_list))
plt.plot(x, loss_list, label='train')
plt.xlabel('iterations (x' + str(eval_interval) + ')')
plt.ylabel('loss')
plt.show()
