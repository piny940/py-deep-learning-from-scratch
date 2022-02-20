import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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

def print_similar_words(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索

    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = get_cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def get_contexts_target(corpus, window_size=1):
    target = corpus[window_size:len(corpus) - window_size]
    contexts = []
    for idx in target:
        c = []
        for j in range(-window_size, window_size+1):
            if j == 0:
                continue
            c.append(corpus[idx + j])
        contexts.append(c)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


# def to_gpu(x):
#     import cupy
#     if type(x) == cupy.ndarray:
#         return x
#     return cupy.asarray(x)

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss = 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl
