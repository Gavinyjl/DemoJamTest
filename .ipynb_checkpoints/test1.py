import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.feature_extraction.text import CountVectorizer as cv

msg = 'dddsss'
print('msg: ', msg)


pd.set_option('display.max_columns', 100)
df = pd.read_csv(r"C:\Users\Gavin Yu\Desktop\IMDB_Top250Engmovies2_OMDB_Detailed.csv")
df.head()

