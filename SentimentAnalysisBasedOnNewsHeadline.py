import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Declaring all the variables 
vocab_size = 1000
embedding_dim = 16
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

# Getting the dataset from github
# Splitting the data into sentences and labels (sarcastic = 0, non-sarcastic = 1)
url = 'https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/blob/master/Sarcasm_Headlines_Dataset.json?raw=true'
df = pd.read_json(url,lines=True)
sentences = [];
labels = [];
for index, data in df.iterrows():
  sentences.append(data['headline'])
  labels.append(data['is_sarcastic'])

# Dividing the complete dataset into training and testing dataset 
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Tokenizing all the words in the training sentences
# most frequent 1000 words will be stored
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Converting every sentence into sequence of toknes
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Need this block to get it to work with TensorFlow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Creating neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Training the neural network model 
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# Plotting the graph between epochs and accuracy of training and testing data 
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend(['training', 'testing'])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Testing the model with different sentences 
sentence = ["reports of movie being good reach area man","behind the scenes of an intricate fbi sting"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))