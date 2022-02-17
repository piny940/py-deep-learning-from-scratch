import os, sys
from matplotlib.pyplot import stem
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.time_layers import TimeRNN, TimeAffine, TimeSoftmaxWithLoss, TimeEmbedding


class SimpleRNNLm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_w = (rn(V, D) / 100).astype('f')
        rnn_wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_w = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_w),
            TimeRNN(rnn_wx, rnn_wh, rnn_b, stateful=True),
            TimeAffine(affine_w, affine_b)
        ]
        
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
