import sys
sys.path.append('..')
from preprocess import preprocess, create_co_matrix, most_similar


text ='Anaconda Navigator is my favorite python env.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('anaconda', word_to_id, id_to_word, C, top=5)
