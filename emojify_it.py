#!/usr/bin/env python
# coding: utf-8

# In[7]:


import emoji
import pandas as pd
import numpy as np
from keras.models import Model, load_model


# In[8]:


emoji_dictionary = {'0': '\u2764\uFE0F',
                    '1': ':baseball:',
                    '2': ':grinning_face_with_big_eyes:',
                    '3': ':disappointed_face:',
                    '4': ':fork_and_knife:',
                    '5': ':hundred_points:',
                    '6': ':fire:',
                    '7': ':face_blowing_a_kiss:',
                    '8': ':chestnut:',
                    '9': ':flexed_biceps:'}


# In[9]:


model = load_model('best_model.h5')


# In[16]:


f = open('glove.6B.50d.txt', encoding='utf-8')
embeddings_index = {}

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    embeddings_index[word] = coefs
    
f.close()

emb_dim = embeddings_index['eat'].shape[0]


# In[17]:


def embedding_output(X):
    maxLen = 10
    embedding_out = np.zeros((X.shape[0], maxLen, emb_dim))
    
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for ij in range(len(X[ix])):
            try:
                embedding_out[ix][ij] = embeddings_index[X[ix][ij].lower()]
            except:
                embedding_out[ix][ij] = np.zeros((50,))
    
    return embedding_out


# In[18]:


def predict_emoji(sentence):
    pred = model.predict_classes(embedding_output(pd.Series([sentence])))
    print(sentence)
    print(emoji.emojize(emoji_dictionary[str(pred[0])]))


# In[ ]:





# In[ ]:




