#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


import os
os.chdir('C:/Udemy/Movie Lens/Movie_Lens')


# In[77]:


MOVIELENS_DIR = 'dat'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'


# In[4]:


# Specify User's Age and Occupation Column
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }


# In[5]:


# Define csv files to be saved into
USERS_CSV_FILE = 'users.csv'
MOVIES_CSV_FILE = 'movies.csv'
RATINGS_CSV_FILE = 'ratings.csv'


# In[6]:


# Read the Ratings File
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), sep='::', 
                    engine='python', encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Set max_userid to the maximum user_id in the ratings
max_userid = ratings['user_id'].drop_duplicates().max()
print('Count of unique users in the dataset : ', max_userid)
# Set max_movieid to the maximum movie_id in the ratings
max_movieid = ratings['movie_id'].drop_duplicates().max()
print('Count of unique movies in the dataset : ', max_movieid)

print(len(ratings), 'ratings loaded')
print(ratings.shape)


# In[7]:


ratings.to_csv(RATINGS_CSV_FILE, sep = '\t', 
               header = True, encoding = 'latin-1',
               columns = ['user_id', 'movie_id', 'rating', 'timestamp'])


# In[8]:


# Read the Users File
users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), sep='::', 
                    engine='python', encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])

print(len(users))
users.head()


# In[9]:


users.to_csv(USERS_CSV_FILE, sep = '\t',
             header = True, encoding = 'latin-1',
             columns = ['user_id', 'gender', 'age', 'occupation', 'zipcode'])


# In[10]:


# Read the Movies File
movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), sep = '::', 
                     engine = 'python', encoding = 'latin-1',
                     names = ['movie_id', 'title', 'genres']
                    )

print(len(movies))
movies.head()


# In[11]:


movies.to_csv(MOVIES_CSV_FILE, sep = '\t',
              header = True, encoding = 'latin-1',
              columns = ['movie_id', 'title', 'genres'])


# In[12]:


ratings = pd.read_csv('ratings.csv', sep = '\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings.head()


# In[13]:


ratings.shape


# In[14]:


users = pd.read_csv('users.csv', sep = '\t', encoding='latin-1', usecols=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
users.head()


# In[15]:


AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }


# In[16]:


users['age_descr'] = users['age'].apply(lambda x : AGES[x])
users['occ_descr'] = users['occupation'].apply(lambda x : OCCUPATIONS[x])


# In[17]:


users.drop(columns=['age', 'occupation'])


# In[18]:


movies = pd.read_csv('movies.csv', sep = '\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
movies.head()


# In[19]:


# Checking if the the datasets contain any null values
ratings.info()


# In[20]:


users.info()


# In[21]:


movies.info()


# In[22]:


# None of the dataset contain any NULL values 


# In[23]:


import wordcloud
from wordcloud import WordCloud, STOPWORDS


# In[24]:


# Generate a word cloud vizualization for Movie Titles

movies['title'] = movies['title'].fillna("").astype('str')
title_corp = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height = 2000, width = 4000).generate(title_corp)

print(title_corp)


# In[25]:



plt.figure(figsize=(30,20))
plt.imshow(title_wordcloud)
plt.axis('off')       # does not show x & y axis
plt.show()


# In[26]:


# From the above word cloud its clearly visible that few commonly used words such as Man, Love, Day, Night, Big, Time, Dead etc.


# In[27]:


genre_labels = set()    # using sets since it does not contain any duplicate values and it will be easier to create a set of all genres.
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
    
print(genre_labels)


# In[28]:


import collections


# In[29]:


def count_word(dataset, ref_col, genre_lab):
    keyword_count = dict()
    for s in genre_lab:
        keyword_count[s]=0
        
    for keyword_genre_label in dataset[ref_col].str.split('|'):
        for s in [s for s in keyword_genre_label if s in genre_lab]:
            keyword_count[s] += 1
            
    keyword_occurence = []
    for k,v in keyword_count.items():
        keyword_occurence.append([k, v])
    keyword_occurence.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurence, keyword_count
    


# In[30]:


keyword_occurence, dummy = count_word(movies, 'genres', genre_labels)
keyword_occurence[:6]


# In[31]:


# Generate a word cloud vizualization for Genres

genres = dict()
for a in keyword_occurence[:]:
    genres[a[0]] = a[1]

print(genres)


# In[32]:


genre_wordcloud = WordCloud(width= 4000, height=2000, background_color='white').generate_from_frequencies(genres)


# In[33]:


plt.figure(figsize=(30,20))
plt.imshow(genre_wordcloud)
plt.axis('off')       # does not show x & y axis
plt.show()


# In[34]:


# Its evident from the above genre wordcloud that Drama, Comedy, Action, Thriller, Romance are the top 5 genres.


# In[ ]:





# In[35]:


movies['genres'] = movies['genres'].str.split('|')

movies['genres'] = movies['genres'].fillna("").astype('str')


# In[36]:


movies['genres']


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer   
# TfidfVectorizer Convert a collection of raw documents to a matrix of TF-IDF features
tf = TfidfVectorizer(analyzer= 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape


# In[38]:


print(tf.get_feature_names())


# In[39]:


# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. 
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. 


# In[40]:


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[41]:


cosine_sim[3]


# In[42]:


# we have now created a pairwise cosine similarity matrix for all movies in the dataset.
# Now we will create a function to return top 20 similar movies based on the cosine similarity score


# In[61]:


titles = movies['title']
indices = pd.Series(movies.index, index = movies['title'])


# In[62]:


# Generating only the top 20 movie recommendations
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[0:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[63]:


genre_recommendations('Good Will Hunting (1997)').head(20)


# In[ ]:





# In[64]:


genre_recommendations('Saving Private Ryan (1998)').head(20)


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


ratings['user_id'] = ratings['user_id'].fillna(0)
ratings['movie_id'] = ratings['movie_id'].fillna(0)


# In[66]:


ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())


# In[67]:


from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(ratings, test_size = 0.2)


# In[68]:


train_data.shape, test_data.shape


# In[69]:


train_data_matrix = train_data.as_matrix(columns=['user_id', 'movie_id', 'rating'])
test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])

train_data_matrix.shape, test_data_matrix.shape


# In[ ]:





# In[70]:


from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])


# In[ ]:




