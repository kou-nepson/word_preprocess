import sys
sys.path.append('..')
import numpy as np
from preprocess import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)

print(id_to_word)

C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0],
    ], dtype=np.int32)
print(C[0])

print(C[4])

print(C[word_to_id['goodbye']])
