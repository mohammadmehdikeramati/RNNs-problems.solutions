# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:26:47 2020

@author: Mohammad Mehdi Keramati
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="3"

from tensorflow import keras
from keras import layers



train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train"
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test"
)


for inputs, targets in train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets)
    break


text_only_train_ds = train_ds.map(lambda x, y: x)

max_length = 600 
max_tokens = 20000 


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




inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
                                                                  
############## Adding bias ####################################
x = layers.GRU(32,bias_initializer="ones",use_bias=True)(embedded)
###############################################################

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", 
              loss="binary_crossentropy",
              metrics=["accuracy"])
              

model.summary()

model.fit(int_train_ds, validation_data=int_test_ds, epochs=10)
