import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.util import processor, get_contexts_target, convert_one_hot
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW


text = 'You say goodbye and I say hello.'
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

corpus, word_to_id, id_to_word = processor(text)
vocab_size = len(word_to_id)

model = SimpleCBOW(vocab_size, hidden_size)

contexts, target = get_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
optimizer = Adam()

trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
