import os

data_dir = '/home/lodeguns/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

#understand the structure and the number of lines
print(header)
print(len(lines))

import numpy as np

#Let's convert all of these 420,551 lines of data into a Numpy array:
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

#plot
from matplotlib import pyplot as plt
temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()
#Here is a more narrow plot of the first ten days of temperature data
#(since the data is recorded every ten minutes, we get 144 data points per day):
plt.plot(range(1440), temp[:1440])
plt.show()
#On this plot, you can see daily periodicity, especially evident for the last 4 days.


#Write a Python generator that takes our current array of float data and yields batches of data
#from the recent past, alongside with a target temperature in the future. Since the samples in
#our dataset are highly redundant (e.g. sample N and sample N + 1 will have most of their timesteps
#in common), it would be very wasteful to explicitly allocate every sample. Instead, we will generate
#the samples on the fly using the original data.
#We preprocess the data by subtracting the mean of each timeseries and dividing by the standard deviation.
#We plan on using the first 200,000 timesteps as training data, so we compute the mean and standard
#deviation only on this fraction of the data:


mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# Now here is the data generator that we will use.
# It yields a tuple (samples, targets) where samples is one batch of input data and  targets is the
# corresponding array of target temperatures. It takes the following arguments:

#data: The original array of floating point data, which we just normalized in the code snippet above.
#lookback: How many timesteps back should our input data go.
#delay: How many timesteps in the future should our target be.
#min_index and max_index: Indices in the data array that delimit which timesteps to draw from.
# This is useful for keeping a segment of the data for validation and another one for testing.
#shuffle: Whether to shuffle our samples or draw them in chronological order.
#batch_size: The number of samples per batch.
#step: The period, in timesteps, at which we sample data. We will set it 6 in order to draw one data point every hour.

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

## Now let's use our abstract generator function to instantiate three generators, one for training,
## one for validation and one for testing. Each will look at different temporal segments of the
## original data: the training generator looks at the first 200,000 timesteps, the validation generator
## looks at the following 100,000, and the test generator looks at the remainder.


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

#A common sense, non-machine learning baseline
#Before we start leveraging black-box deep learning models to solve our temperature prediction problem,
#  let's try out a simple common-sense approach. It will serve as a sanity check, and it will establish a
# baseline that we will have to beat in order to demonstrate the usefulness of more advanced machine learning models.
# Such common-sense baselines can be very useful when approaching a new problem for which there is no known solution
# (yet). A classic example is that of unbalanced classification tasks, where some classes can be much more common than
# others. If your dataset contains 90% of instances of class A and 10% of instances of class B, then a common sense
# approach to the classification task would be to always predict "A" when presented with a new sample.
# Such a classifier would be 90% accurate overall, and any learning-based approach should therefore beat this 90%
# score in order to demonstrate usefulness. Sometimes such elementary baseline can prove surprisingly hard to beat.


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
    preds = samples[:, -1, 1]
    mae = np.mean(np.abs(preds - targets))
    batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()





#A basic machine learning approach
#In the same way that it is useful to establish a common sense baseline before trying machine
# learning approaches, it is useful to try simple and cheap machine learning models
#(such as small densely-connected networks) before looking into complicated and computationally
# expensive models such as RNNs. This is the best way to make sure that any further complexity we
# throw at the problem later on is legitimate and delivers real benefits.

#Here is a simply fully-connected model in which we start by flattening the data, then run it
# through two Dense layers. Note the lack of activation function on the last Dense layer, which
#is typical for a regression problem. We use MAE as the loss. Since we are evaluating on the exact
#  same data and with the exact same metric as with our common sense approach,
# the results will be directly comparable.

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#A first recurrent baseline¶
#Our first fully-connected approach didn't do so well, but that doesn't mean machine
## learning is not applicable to our problem. The approach above consisted in first
## flattening the timeseries, which removed the notion of time from the input data.
# Let us instead look at our data as what it is: a sequence, where causality and order matter.
#  We will try a recurrent sequence processing model -- it should be the perfect fit for
#  such sequence data, precisely because it does exploit the temporal ordering of data
#  points, unlike our first approach.

#Instead of the LSTM layer introduced in the previous section, we will use the GRU layer,
#  developed by Cho et al. in 2014. GRU layers (which stands for "gated recurrent unit")
#  work by leveraging the same principle as LSTM, but they are somewhat streamlined and
# thus cheaper to run, albeit they may not have quite as much representational power as LSTM.
# This trade-off between computational expensiveness and representational power is seen
# everywhere in machine learning.

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#Using recurrent dropout to fight overfitting
#It is evident from our training and validation curves that our model is overfitting:
# the training and validation losses start diverging considerably after a few epochs. You are already
#  familiar with a classic technique for fighting this phenomenon: dropout, consisting in randomly
#  zeroing-out input units of a layer in order to break happenstance correlations in the training
# data that the layer is exposed to. How to correctly apply dropout in recurrent networks, however,
#  is not a trivial question. It has long been known that applying dropout before a recurrent layer
#  hinders learning rather than helping with regularization. In 2015, Yarin Gal, as part of his Ph.D.
#  thesis on Bayesian deep learning, determined the proper way to use dropout with a recurrent network:
# the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead
#  of a dropout mask that would vary randomly from timestep to timestep. What's more: in order to
# regularize the representations formed by the recurrent gates of layers such as GRU and LSTM, a
# temporally constant dropout mask should be applied to the inner recurrent activations of the layer
# (a "recurrent" dropout mask). Using the same dropout mask at every timestep allows the network to
#  properly propagate its learning error through time; a temporally random dropout mask would instead
#  disrupt this error signal and be harmful to the learning process.
# Yarin Gal did his research using Keras and helped build this mechanism directly
#  into Keras recurrent layers. Every recurrent layer in Keras has two dropout-related
# arguments: dropout, a float specifying the dropout rate for input units of the layer, and
# recurrent_dropout, specifying the dropout rate of the recurrent units. Let's add dropout and
#  recurrent dropout to our GRU layer and see how it impacts overfitting. Because networks being
#  regularized with dropout always take longer to fully converge, we train our network for twice as many epochs.

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#Using recurrent dropout to fight overfitting
#It is evident from our training and validation curves that our model is overfitting:
# the training and validation losses start diverging considerably after a few epochs.
# You are already familiar with a classic technique for fighting this phenomenon: dropout,
#  consisting in randomly zeroing-out input units of a layer in order to break happenstance
# correlations in the training data that the layer is exposed to. How to correctly apply
#  dropout in recurrent networks, however, is not a trivial question. It has long been
#  known that applying dropout before a recurrent layer hinders learning rather than helping
#  with regularization. In 2015, Yarin Gal, as part of his Ph.D. thesis on Bayesian deep
# learning, determined the proper way to use dropout with a recurrent network: the same dropout
#  mask (the same pattern of dropped units) should be applied at every timestep, instead of a
# dropout mask that would vary randomly from timestep to timestep. What's more: in order to
# regularize the representations formed by the recurrent gates of layers such as GRU and LSTM,
#  a temporally constant dropout mask should be applied to the inner recurrent activations of
#  the layer (a "recurrent" dropout mask). Using the same dropout mask at every timestep allows
# the network to properly propagate its learning error through time; a temporally random dropou
# t mask would instead disrupt this error signal and be harmful to the learning process.Yarin
#  Gal did his research using Keras and helped build this mechanism directly into Keras recurrent
#  layers. Every recurrent layer in Keras has two dropout-related arguments: dropout, a float
#  specifying the dropout rate for input units of the layer, and recurrent_dropout, specifying
# the dropout rate of the recurrent units. Let's add dropout and recurrent dropout to our GRU
# layer and see how it impacts overfitting. Because networks being regularized with dropout always
#  take longer to fully converge, we train our network for twice as many epochs.


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Stacking recurrent layers
#Since we are no longer overfitting yet we seem to have hit a performance bottleneck,
#  we should start considering increasing the capacity of our network. If you remember
#  our description of the "universal machine learning workflow": it is a generally a good
# idea to increase the capacity of your network until overfitting becomes your primary
# obstacle (assuming that you are already taking basic steps to mitigate overfitting, s
# uch as using dropout). As long as you are not overfitting too badly, then you are likely
#  under-capacity.Increasing network capacity is typically done by increasing the number
#  of units in the layers, or adding more layers. Recurrent layer stacking is a classic
# way to build more powerful recurrent networks: for instance, what currently powers the
# Google translate algorithm is a stack of seven large LSTM layers -- that's huge.To stack
#  recurrent layers on top of each other in Keras, all intermediate layers should return
#  their full sequence of outputs (a 3D tensor) rather than their output at the last timestep
# . This is done by specifying return_sequences=True:

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()