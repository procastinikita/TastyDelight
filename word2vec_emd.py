import numpy as np # linear algebra
import pandas as pd 
import nltk
import string
import ast
import re
import unidecode

from parser_1 import ingredient_parser

# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
from gensim.models import Word2Vec
import logging

from gensim.models import Word2Vec

# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.ingredients_parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

# load in data
data = pd.read_csv('/home/anya/major_project/dataset/Indian_Recipe_dataset.csv')
# parse the ingredients for each recipe
data['ingredients_parsed'] = data.TranslatedIngredients.apply(ingredient_parser)
# get corpus
corpus = get_and_sort_corpus(data)
print(f"Length of corpus: {len(corpus)}")
# calculate average length of each document 
lengths = [len(doc) for doc in corpus]
avg_len = float(sum(lengths)) / len(lengths)
# train and save CBOW Word2Vec model
# train word2vec model 
sg = 0 # CBOW: build a language model that correctly predicts the center word given the context words in which the center word appears
workers = 8 # number of CPUs
window = 4 # window size: average length of each document 
min_count = 1 # unique ingredients are important to decide recipes 

model_cbow = Word2Vec(corpus, sg=sg, workers=workers, window=4, min_count=min_count, vector_size=100)

model_cbow.save('models/model_cbow.bin')