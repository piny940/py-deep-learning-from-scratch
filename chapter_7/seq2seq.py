import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.time_layers import TimeEmbedding, TimeLSTM
import numpy as np


class Encoder:
    def __init__(self, vocab_size, word_vec_size, hidden_size):
        V, D, H = vocab_size, word_vec_size, hidden_size
        rn = np.random.randn

        embed_w = (rn(V, D) / 100).astype('f')
        lstm_wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        
        self.embed = TimeEmbedding(embed_w)
        self.lstm = TimeLSTM(lstm_wx, lstm_wh, lstm_b)
        
        self.params = [self.embed.params + self.lstm.params]
        self.grads = [self.embed.grads + self.lstm.grads]
        self.hs = None
    
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
