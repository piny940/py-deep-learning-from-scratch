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
