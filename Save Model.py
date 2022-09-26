# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 03:02:55 2022

@author: Mohammad Mehdi Keramati
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="3"

from tensorflow import keras
from keras import layers

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)



text_only_train_ds = train_ds.map(lambda x, y: x)



max_length = 6 #600
max_tokens = 200 #20000


text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)




text_vectorization.adapt(text_only_train_ds)



################## Model #############################

inputs = keras.Input(shape=(1,), dtype="string")
vectorized=text_vectorization(inputs) 
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(vectorized)
x = layers.LSTM(32)(embedded) 
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)    

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
              
model.summary()


model.fit(train_ds, validation_data=test_ds, epochs=10)

#################### Saving Model ######################

model.save("my_model")

#################### Load Model ########################

reconstructed_model = keras.models.load_model("my_model")



for inputs, targets in train_ds:
    
    print("prediction[0]:", inputs[0])
    print("prediction[0]:", reconstructed_model.predict(inputs[0]))
    print("targets[0]:", targets[0])
    
    
    break





