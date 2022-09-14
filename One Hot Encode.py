# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 23:58:55 2022

@author: Mohammad Mehdi Keramati
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="3"

from tensorflow import keras
from keras import layers
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing


encode = OneHotEncoder()

batch_size = 1

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train"
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)



max_length = 600 
max_tokens = 20000 



text_only_train_ds = train_ds.map(lambda x, y: x)


text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)



text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)


for inputs, targets in int_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    
    print('###############################')
    print("inputs[0]:", inputs[0])
    print('###############################')
    print("targets[0]:", targets)
    break

####################### One hot encoding ############################
int_train_ds = int_train_ds.map(
    lambda x, y: (int(tf.one_hot(x, depth= 256, axis=-1)), int(y)),
    num_parallel_calls=4)

int_test_ds = int_test_ds.map(
    lambda x, y: (int(tf.one_hot(x, depth=256, axis=-1)), int(y)),
    num_parallel_calls=4)
####################################################################

for inputs, targets in int_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    
    print('###############################')
    print("inputs[0]:", inputs[0])
    print('###############################')
    print("targets[0]:", targets)
    break




inputs = keras.Input(shape=(600, 256), dtype="int32") ########## Still I have problem here ???? #########
x = layers.GRU(32,bias_initializer="ones",use_bias=True)(inputs)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", #
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()

model.fit(int_train_ds, epochs=10)


