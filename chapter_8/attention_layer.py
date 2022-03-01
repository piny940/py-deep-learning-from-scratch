import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.layers import Softmax


class WeightSum:
    def __init__(self):
        self.params = []
        self.grads = []
        self.cache = None
    
    def forward(self, hs, a):
        N, T, H = hs.shape
        
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)
        
        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params = []
        self.grads = []
        self.softmax = Softmax()
        self.cache = None
    
    def forward(self, hs, h):
        N, T, H = hs.shape
        
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        
        a = self.softmax.forward(s)
        self.cache = (hs, hr)
        
        return a
    
    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape
        
        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis = 1)
        return dhs, dh


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_layer = Attention()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None
    
    def forward(self, hs, h):
        a = self.attention_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh
