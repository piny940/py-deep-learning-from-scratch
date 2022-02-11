import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        v, h = vocab_size, hidden_size

        w_in = 0.01 * np.random.randn(v, h).astype('f')
        w_out = 0.01 * np.random.randn(h, v).astype('f')
        
        self.in_layer0 = MatMul(w_in)
        self.in_layer1 = MatMul(w_in)
        self.out_layer = MatMul(w_out)
        self.loss_layer = SoftmaxWithLoss()
        
        layers = [
            self.in_layer0,
            self.in_layer1,
            self.out_layer,
            self.loss_layer
        ]
        self.params, self.grads = [], []
        
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = w_in
    
    def forward(self, contexts, target):
        h0 = contexts[:, 0]
        h1 = contexts[:, 1]
        h0 = self.in_layer0.forward(h0)
        h1 = self.in_layer1.forward(h1)
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        
        return None
