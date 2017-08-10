import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series)- window_size):
        X.append(series[i : i + window_size])

    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    # LSTM layer with 5 hidden nodes
    model.add(LSTM(5, input_shape=(window_size, 1)))

    # fully connected layer
    model.add(Dense(1))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):

    punctuation = ['!', ',', '.', ':', ';', '?']

    ascii_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
             ]

    allowed_chars = punctuation + ascii_chars + [' ']

    unique_chars = sorted(list(set(text)))

    for c in unique_chars:

        if c not in allowed_chars:
            # remove un-needed char
            text = text.replace(c, '')

    return text



### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    start = 0
    end = len(text) - window_size

    while end >= start:
        inputs.append(text[start: start + window_size])
        outputs.append(text[start + window_size])
        start += step_size

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):

    model = Sequential()

    # LSTM layer with 200 hidden nodes
    model.add(LSTM(200, input_shape=(window_size, num_chars)))

    # fully connected layer
    model.add(Dense(num_chars))

    model.add(Activation('softmax'))

    return model

