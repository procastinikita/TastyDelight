import pandas as pd
import numpy as np
import nltk
import ast
import re
import string 
from nltk.stem import WordNetLemmatizer
from nltk import corpus
import unidecode
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# nltk.download('wordnet')

def ingredient_parser(ingredients):
    # measures and common words (already lemmatized)   
    measures = ['teaspoon', 'cup', 'tsp', 'tablespoon','tablespoons','gm', 'T','inch','liter','gram','grams','spoon','tbsp','kg','length','pound','bottle']
    words_to_remove=['powder', 'chopped', 'salt', 'chilli', 'finely', 'taste', 'leaf', 'oil', 'red','green','eyed','fine','round','pistachio','pearl',
  'onion', 'garlic', 'sunflower', 'cumin', 'turmeric', 'clove', 'ginger', 'grated', 'whole', 'water', 'sugar', 'cut', 'dry', 'sliced','ripe','spice',
'fresh', 'required', 'asafoetida', 'pinch', 'olive', 'per', 'paste', 'extra', 'virgin', 'sauce', 'small', 'soaked', 'peeled', 'methi', 'stick', 'boiled',
 'diced', 'dried', 'pod', 'roasted', 'thinly', 'baking','seed','deseeded','coriander','pepper','sprig','masala','rolled','purpose','garam','paprika','break',
'crushed', 'cooking', 'fennel', 'piece', 'garnish', 'roughly', 'slit', 'minced','mustard','like','salted','full','square','made','dusting','boil','stem','gingelly',
'cube', 'extract', 'flake', 'pea', 'needed', 'spinach', 'juice','turmeric','leaves','cardamom','ajwain','elaichi','cinnamon','mix','break','rose','serve','lemongrass',
 'use', 'puree', 'homemade', 'powdered', 'frying', 'spring', 'raw', 'cooked', 'hing','sweet','baking','ka','used','ssd','root','hard','oregano','dusting','full','half',
'basil', 'mixed', 'vinegar', 'kashmiri', 'bunch', 'washed', 'amchur', 'vanilla','cashew','cumin','big','half','grind','extracted','drop','britannia',
'ground', 'saffron', 'strand', 'button', 'cubed', 'thin', 'instant', 'deep', 'ml','sugar','bell','nut','tighten','fried','soda','poppy','anise','phoran',
 'chili', 'coarsely', 'caster', 'hung', 'fruit', 'broken', 'hour', 'torn','dried','tightly','packed','cardamom','chilies','squeezed','fat','toasted','canned',
 'optional', 'mashed', 'vegetable', 'parsley', 'kasuri', 'overnight', 'pounded', 'sesame','garam masala','pitted','softened','quartered','nutralite','pan',
'chop', 'hot', 'half', 'freshly', 'shredded', 'curry','removed','reserved','boiling','cardamom crushed','black pepper','adjustablt','thick','spice',
 'chunk', 'basmati', 'lukewarm', 'long', 'stock', 'slice', 'nutmeg', 'crumb', 'fry', 'chilled', 'black peppercorn crushed','condensed','pitted','original','lime',
'minute', 'halved', 'seasoning', 'steamed', 'baby', 'handful', 'melted', 'active', 'pureed', 'crush','radish','cleaned','pickled','whishked','iceberg',
  'cook', 'lengthwise', 'date', 'skin', 'star', 'zest', 'purpose flour','warm','lemon','curd','cheesy','mix','icing','panch','bite','fried','asafetida',
 'crumbled', 'peel', 'strip', 'sized', 'blanched', 'size', 'spread', 'garnishing','plus','roast','heavy','take','yum','rock','sour','well','lightly','grind',
 'little', 'straight', 'according','fig','take','cold','bay','mint','honey','britannia','classic','large','choice','greasing','wash','adjust','fig','serving','spicy','drained',]       
    # Turn ingredient list from string into a list
    if isinstance(ingredients, list):
        ingredients = ingredients
    else:
        ingredients = ingredients.split(",")
    # We first get rid of all the punctuation
    translator = str.maketrans('', '', string.punctuation)
    # initialize nltk's lemmatizer    
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        #removing brackets
        i=re.sub("\(.*?\)","",i)
        i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(' |-|/', i)
        # Get rid of words containing non alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
        # remove accents
        items = [unidecode.unidecode(word) for word in items]
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # get rid of stop words
        stop_words = set(corpus.stopwords.words('english'))
        items = [word for word in items if word not in stop_words]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        items = [word for word in items if word not in measures]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(' '.join(items))
    return ingred_list