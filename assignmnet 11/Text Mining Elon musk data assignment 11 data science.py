#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1) Perform sentimental analysis on the Elon-musk tweets


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


get_ipython().system('pip install stopwords')


# In[5]:


get_ipython().system('pip install textblob')


# In[6]:


from nltk.corpus import stopwords


# In[23]:


from textblob import TextBlob


# In[34]:


data=pd.read_csv('C:/Users/pranjali/Downloads/Elon_musk.csv',encoding="latin-1")


# In[35]:


data.head(10)


# In[36]:


#Number of Words in single tweet
data['word_count'] = data['Text'].apply(lambda x: len(str(x).split(" ")))
data[['Text','word_count']].head(10)


# In[37]:


#Number of characters in single tweet
data['char_count'] = data['Text'].str.len()
data[['Text','char_count']].head(10)


# In[38]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['Text'].apply(lambda x: avg_word(x))
data[['Text','avg_word']].head(10)


# In[39]:


#number of stop words
import nltk
nltk.download('stopwords')

stop = stopwords.words('english')

data['stopwords'] = data['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['Text','stopwords']].head(10)


# In[40]:


#number of special characters
data['hastags'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
data[['Text','hastags']].head(10)


# In[41]:


# no of numerical values
data['numerics'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['Text','numerics']].head(10)


# In[42]:


data['upper'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['Text','upper']].head(10)


# In[ ]:


pre-processing


# In[44]:


data['Text'] = data['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Text'].head()


# In[45]:


#removing punctuation 
data['Text'] = data['Text'].str.replace('[^\w\s]','')
data['Text'].head()


# In[46]:


#removing stop words
stop = stopwords.words('english')
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Text'].head()


# In[47]:


#removing common words
freq = pd.Series(' '.join(data['Text']).split()).value_counts()[:10]
freq


# In[48]:


freq = list(freq.index)
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Text'].head()


# In[49]:


#removing rare words
freq = pd.Series(' '.join(data['Text']).split()).value_counts()[-10:]
freq


# In[50]:


freq = list(freq.index)
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Text'].head()


# In[51]:


data['Text'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[52]:


import nltk
nltk.download('punkt')

TextBlob(data['Text'][1]).words


# In[54]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[55]:


from textblob import Word

import nltk
nltk.download('wordnet')

data['Text'] = data['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Text'].head()


# In[56]:


TextBlob(data['Text'][0]).ngrams(2)


# In[57]:


tf1 = (data['Text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# In[ ]:


Inverse Document Frequency IDF=log(N/n), where, N is the total number of rows and n is the number of rows in
which the word was present.


# In[58]:


for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Text'].str.contains(word)])))

tf1


# In[ ]:


Term Frequency-Inverse Document Frequency (TF-IDF)


# In[59]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# In[60]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(data['Text'])
vect


# In[ ]:


Bag of words


# In[61]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Text'])
data_bow


# In[ ]:


Sentiment Analysis


# In[62]:


data['Text'][:5].apply(lambda x: TextBlob(x).sentiment)


# In[63]:


data['sentiment'] = data['Text'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Text','sentiment']].head(10)


# In[ ]:




