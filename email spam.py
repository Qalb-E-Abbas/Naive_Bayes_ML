# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:36:11 2021

@author: Qalbe
"""

import pandas as pd

df = pd.read_csv("spam.csv")
df.head()

# check all, how many spams and ham and others

df.groupby('Category').describe()


# convert spam t0 numbers
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam)


# representing words as count for msgs
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]


#naive_bayes has three types, each one for different function. This one for movie rating etc

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)



emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)

model.predict(emails_count)