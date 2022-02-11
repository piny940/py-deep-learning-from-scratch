import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingWithLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        v, h = vocab_size, hidden_size

        w_in = 0.01 * np.random.randn(v, h).astype('f')
        w_out = 0.01 * np.random.randn(h, v).astype('f')
        
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(w_in)
            self.in_layers.append(layer)
        
        self.ns_loss = NegativeSamplingWithLoss(w_out, corpus)
        
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
    
        self.word_vecs = w_in
    
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
