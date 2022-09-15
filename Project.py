# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:47:48 2022

@author: Mohammad Mehdi Keramati
"""

import csv
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import preprocessing
from imblearn.under_sampling import RandomUnderSampler

########################### Preprocessing ######################################

Data=pd.read_csv('amazon_baby.csv') # importing dataset


Data=Data.dropna(axis=0) 
Data=Data.dropna().reset_index(drop=True) # drop rows with Nan valuse
count=Data.isnull().sum().sum()

Data=Data.drop(columns='name') # drop excess columns


for i in range(0,len(Data)): # convert to two class 
    
 if Data.loc[i]['rating']<3:
     
     Data.at[i,'rating']=0
    
 else: 
     Data.at[i,'rating']=1


################################ Class balancing ##############################

X= np.array(Data.loc[:,'review'])
y= np.array(Data.loc[:,'rating'])


rus = RandomUnderSampler(random_state=42, replacement=True)

X=pd.DataFrame(X)
y=pd.DataFrame(y)

X_rus, y_rus = rus.fit_resample(X, y)

########################## Spliting to train and test #########################

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X, y, random_state=1, train_size = .75, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, random_state=1, train_size = .75, stratify=y_rus)

############################## Vectorization ##################################

max_length = 60 #600
max_tokens = 2000 #20000


text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)


text_vectorization.adapt(X_train)


X_train=text_vectorization(X_train)
X_test=text_vectorization(X_test)

########################### Network architecture ##############################

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.LSTM(32,bias_initializer="ones",use_bias=True)(embedded) 
outputs = layers.Dense(1, activation="sigmoid")(x) # sigmoid
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", 
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

model.fit(X_train, y_train, epochs=10)

######################### Prediction ##########################################

predictions = model.predict(X_test)


test_1=predictions
test_2=predictions.argmax(1)
test_3=np.round(predictions)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,test_3))



