import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size=5):
    # containers for input/output pairs
    X = []
    y = []

    for x in range(0, len(series) - window_size):
        buff = np.array(series[x:x+window_size])
        X.append(buff)
        y.append(np.array([series[x+window_size]]))

    X = np.array(X)
    y = np.array(y)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), return_sequences=False))
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.split(' ')
    text = [t for t in text if '\\' not in t]
    text = ' '.join(text)

    import string

    for p in string.punctuation:
        if p not in punctuation:
            text = text.replace(p, ' ')
        

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for x in range(0, len(text) - window_size - 1, step_size):
        buff = text[x:x+5]
        inputs.append(buff)
        outputs.append(text[x+5])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation="softmax"))
    return model
