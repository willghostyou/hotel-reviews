#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# read data
reviews_df = pd.read_csv("tripadvisor_hotel_reviews.csv")

reviews_df.head()


# In[2]:


# loading libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as L
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from wordcloud import WordCloud 
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sb


import numpy as np 

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import random as rn

import re


# In[3]:


seed_value = 1337
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
rn.seed(seed_value)
print('rows data: {}'.format(len(reviews_df)))
print('any null values?')
reviews_df.isna().sum()


# ### distribution of class

# In[4]:


class_dist = reviews_df['Rating'].value_counts()

def ditribution_plot(x,y,name):
    fig = go.Figure([
        go.Bar(x=x, y=y)
    ])

    fig.update_layout(title_text=name)
    fig.show()
    
ditribution_plot(x= class_dist.index, y= class_dist.values, name= 'Class Distribution')


# ### most used phrases

# In[5]:


def wordCloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud)
    plt.show()

wordCloud_generator(reviews_df['Review'])


# In[6]:


X = reviews_df['Review'].copy()
y = reviews_df['Rating'].copy()


# ## cleaning data
# 

# In[7]:




def data_cleaner(review):    
    # remove digits
    review = re.sub(r'\d+',' ', review)
    
    #removing stop words
    review = review.split()
    
    #review = " ".join([word for word in review if not word in sw])
    
    #Stemming
    #review = " ".join([ps.stem(w) for w in review])
    
    return review

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer() 

sw = stopwords.words('english')

X_cleaned = X.apply(data_cleaner)
X_cleaned.head()


# ### tokenizie

# In[8]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_cleaned)

X = tokenizer.texts_to_sequences(X_cleaned)

max_length = max([len(x) for x in X])
vocab_size = len(tokenizer.word_index)+1
exp_sen = 1

print("Vocabulary size: {}".format(vocab_size))
print("max length of sentence: {}".format(max_length))
print("\nExample:\n")
print("Sentence:\n{}".format(X_cleaned[exp_sen]))
print("\nAfter tokenizing :\n{}".format(X[exp_sen]))

X = pad_sequences(X, padding='post', maxlen=350)
print("\nAfter padding :\n{}".format(X[exp_sen]))


# In[9]:


encoding = {1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4
           }

labels = ['1', '2', '3', '4', '5']
           
y = reviews_df['Rating'].copy()
y.replace(encoding, inplace=True)


# ## spliting and modelling

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=67, stratify=y
)

# hyper parameters
EPOCHS = 3
BATCH_SIZE = 100
embedding_dim = 16
units = 76

model = tf.keras.Sequential([
    L.Embedding(vocab_size, embedding_dim, input_length=X.shape[1]),
    L.Bidirectional(L.LSTM(units,return_sequences=True)),
    #L.LSTM(units,return_sequences=True),
    L.Conv1D(64,3),
    L.MaxPool1D(),
    L.Flatten(),
    L.Dropout(0.5),
    L.Dense(128, activation="relu"),
    L.Dropout(0.5),
    L.Dense(64, activation="relu"),
    L.Dropout(0.5),
    L.Dense(5, activation="softmax")
])


model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer='adam',metrics=['accuracy']
             )

model.summary()


# In[11]:


history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.12, batch_size=BATCH_SIZE, verbose=2)


# In[13]:


pred = model.predict_classes(X_test)
print('Accuracy: {}'.format(accuracy_score(pred, y_test)))
print("Root mean square error: {}".format(np.sqrt(mean_squared_error(pred,y_test))))


# In[14]:


print(classification_report(y_test, pred, target_names=labels))


# In[ ]:




