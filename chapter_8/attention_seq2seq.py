import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from chapter_7.seq2seq import Encoder


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs
    
    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
