from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 10000

# Cut texts after this number of words
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


#Turn back a review from numbers to words (the top 1000).
#word_index = imdb.get_word_index()
#reverse_word_index = dict(
#[(value, key) for (key, value) in word_index.items()])
#decoded_review = ' '.join(
#[reverse_word_index.get(i - 3, '-> ->') for i in x_train[0]])


# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


from keras.layers import Embedding
## The Embedding layer takes at least two arguments:
## the number of possible tokens, here 1000 (1 + maximum word index),
## and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs

#Embedding
#Input shape
#2D tensor with shape: (batch_size, sequence_length).
#Output shape
#3D tensor with shape: (batch_size, sequence_length, output_dim).


model.add(Embedding(10000, 8, input_length=maxlen))
# After the Embedding layer,
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

