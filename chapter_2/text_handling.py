import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.util import processor, get_co_matrix, get_cos_similarity, get_ppmi
import numpy as np
from matplotlib import pyplot as plt

text = 'You say goodbye and I say hello.'

# text = text.lower()
# text = text.replace('.', ' .')
# print(text)

# words = text.split(' ')
# print(words)

corpus, word_to_id, id_to_word = processor(text)
co_matrix = get_co_matrix(corpus)
ppmi = get_ppmi(co_matrix)
U, S, D = np.linalg.svd(ppmi)
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0], U[word_id,1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
