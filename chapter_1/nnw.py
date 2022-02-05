
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/ansai/Documents/VScode/deep-learning/deep-learning-from-scratch')
from dataset import spiral

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
    
    def get_softmax(self, x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))

        return x

    def forward(self, x, t):
        self.t = t
        self.y = self.get_softmax(x)
        if self.y.ndim == 1:
            self.t = self.t.reshape(1, self.t.size)
            self.y = self.y.reshape(1, self.y.size)
            
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
                
        if self.y.ndim == 1:
            self.t = self.t.reshape(1, self.t.size)
            self.y = self.y.reshape(1, self.y.size)
            
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
                
        batch_size = self.y.shape[0]

        return -np.sum(np.log(self.y[np.arange(batch_size), self.t] + 1e-7)) / batch_size
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        
        return dx
