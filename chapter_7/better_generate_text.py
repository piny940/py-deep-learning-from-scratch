import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rnn_gen import BetterRnnlmGen
from dataset import ptb
import numpy as np


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen()

model.load_params('./chapter_6/BetterRnnlm.pkl')

# start_word = 'you'
# start_id = word_to_id[start_word]
skip_words = [
    'N',
    '<unk>',
    '$'
]

skip_ids = [word_to_id[word] for word in skip_words]

start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)
