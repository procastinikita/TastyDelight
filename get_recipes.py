from gensim.models import Word2Vec
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Recommendation import get_recommendations
from collections.abc import Mapping

from Tfidf_emd import TfidfEmbeddingVectorizer
from mean_embedding import MeanEmbeddingVectorizer
from parser_1 import ingredient_parser
#, defaultdict


def get_recs(ingredients,N=5,mean=False):
    """
    Get the top N recipe recomendations.
    :param ingredients: comma seperated string listing ingredients
    :param N: number of recommendations
    :param mean: False if using tfidf weighted embeddings, True if using simple mean
    """
    # load in word2vec model
    #model = Word2Vec.load("models/model_cbow.bin")
    model = pickle.load(open('models/word2vec_model.pkl', 'rb'))
    # normalize embeddings
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")
    # load in data
    #data = pd.read_csv("input/d
    # parse ingredients
    #data["parsed"] = data.ingredients.apply(ingredient_parser)
    # create corpus
    #corpus = get_and_sort_corpus(df)

    if mean:
        """
        #get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus) """
        doc_vec=pickle.load(open('models/mean_emd.pkl', 'rb'))
    else:
        """
     # use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)"""
        doc_vec=pickle.load(open('models/tf_idf.pkl', 'rb'))

     # create embeddings for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    # parse ingredient list
    input = ingredient_parser(input)
    # get embeddings for ingredient doc
    if mean:
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit([input])
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations
