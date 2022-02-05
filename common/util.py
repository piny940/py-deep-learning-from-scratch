import numpy as np


def processor(text):
    text = text.lower()
    text = text.replace('.', ' .')
    text = text.replace(',', ' ,')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word
    
    corpus = [word_to_id[word] for word in words]
    return corpus, word_to_id, id_to_word

def get_co_matrix(corpus, window_size=1):
    vocab_size = np.max(corpus) + 1
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for i, id in enumerate(corpus):
        for j in range(1, window_size+1):
            left_idx = i - j
            right_idx = i + j
            if left_idx >= 0:
                left_id = corpus[left_idx]
                co_matrix[id][left_id] += 1
            if right_idx < corpus_size:
                right_id = corpus[right_idx]
                co_matrix[id][right_id] += 1
    return co_matrix

def get_cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.sqrt(np.sum(x**2))+eps) / (np.sqrt(np.sum(y**2))+eps)

def get_ppmi(co_matrix, verbose=False, eps=1e-8):
    ppmi = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix)
    progress = 0
    total = co_matrix.shape[0] * co_matrix.shape[1]
    count = np.sum(co_matrix, axis=0)
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(co_matrix[i, j] * N / (count[i] * count[j]) + eps)
            ppmi[i, j] = max(0, pmi)

            if verbose:
                progress += 1
                if progress % (total // 100) == 0:
                    print(f'{100*progress/total} done')
    return ppmi