import numpy as np

#number of time steps in the input sequence
timesteps = 100
#dimensionality of the input feature space
input_features = 32
#dimensionality of the output feature space
output_features = 64

#Input data: random noise for the sake of the example (100 arrays of 32 elements)
inputs = np.random.random((timesteps, input_features))

#Array of 64 zero elements
state_t = np.zeros((output_features,))

# the transformation of the input and state into an
# output will be parameterized by two matrices, W and U , and a bias vector
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))


successive_outputs = [] #void list
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) #activation function on input/state t
    successive_outputs.append(output_t)     #Stores this output in a list
    state_t = output_t                      #Updates the state of the network for the next timestep

#The final output is a 2D tensor of shape (timesteps, output_features).
final_output_sequence = np.concatenate(successive_outputs, axis=0)

print(final_output_sequence)