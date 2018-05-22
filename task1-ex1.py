import keras
import numpy as np
print(keras.__version__)


# This is our initine entry per "sample"
# (in this toy example, a "sample" is just a sentence, but
# it could be an entire document).
samples = ['around the world and back again', 'back again from the world']

# First, build an index of all tokens in the data.
token_index = {}
for sample in samples:
    # We simply tokenize the samples via the `split` method.
    # in real life, we would also strip punctuation and special characters
    # from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            print(token_index)
            # Note that we don't attribute index 0 to anything.

# Next, we vectorize our samples.
# We will only consider the first `max_length` words in each sample.
max_length = 7

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

print(token_index)
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        print(word)
        print(i)
        print(j)
        print(index)
        results[i, j, index] = 1.
        print(results[i,])
        print("...")
