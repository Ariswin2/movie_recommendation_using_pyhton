import pandas as pd
import ast
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Read the datasets
credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")

# Display options for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(credits_df.head())
print(movies_df.head())
movies_df = movies_df.merge(credits_df, on='title')
print(movies_df.shape)
print(movies_df.head())
print(movies_df.info())
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
print(movies_df.head())
print(movies_df.info())

print(movies_df.isnull().sum())
movies_df.dropna(inplace=True)
print(movies_df.isnull().sum())

print(movies_df.duplicated().sum())

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
print(movies_df.head())
print(movies_df.info())

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies_df['cast'] = movies_df['cast'].apply(convert3)
movies_df['crew'] = movies_df['crew'].apply(fetch_director)
print(movies_df.head())

movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
new_df = movies_df[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
print(new_df.head())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors.shape)

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)
print(similarity.shape)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('The Avengers')