import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Reshape, Dropout, SimpleRNN
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import os
import tensorflow as tf
import keras as k

def baseline_model():

    chroma = Input(shape=(12, 200, 1))
    mfccs = Input(shape=(20, 200, 1))
    rolloff = Input(shape=(200, )) # (200,)
    centroid = Input(shape=(200, ))
    timbre = Input(shape=(10,))
    tempo = Input(shape=(1,))

    # first branch
    x = Conv2D(filters=4, input_shape=(12, 200, 1), kernel_size=(3, 3), activation='relu')(chroma)
    x = MaxPooling2D((2,2))(x)
    x = Reshape((1980,))(x)
    x = Dense(100, activation='relu')(x)
    # x = Dropout(0.2)(x)

    # second branch
    y = Conv2D(filters=4, input_shape=(20, 200, 1), kernel_size=(3, 3), activation='relu')(mfccs)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Reshape((3564,))(y)
    y = Dense(100, activation='relu')(y)

    # third branch
    z = Dense(100, activation='relu')(centroid)
    z = Dense(40, activation='relu')(z)
    z = Dense(20, activation='relu')(z)

    # fourth branch
    a = Dense(100, activation='relu')(rolloff)
    a = Dense(20, activation='relu')(a)

    # fifth branch
    b = Dense(10, activation='relu')(timbre)

    # sixth branch
    # c = Dense(1, activation='relu')(tempo)

    print(x.shape, y.shape, z.shape, a.shape, b.shape) #, c.shape)

    # combined = np.concatenate([x.output, y.output, z.output, a.output, b.output, c.output])
    combined = Concatenate()([x, y, z, a, b]) # add c if want

    # final layer
    final = Dense(100, activation='relu')(combined)
    final = Dense(20, activation='relu')(final)
    final = Dense(1)(final)

    model = Model(inputs=[chroma, mfccs, rolloff, centroid, timbre], outputs=final) #add tempo here
    model.compile(loss='mean_squared_error', optimizer=k.optimizers.adadelta(lr=1.0, rho=0.95), metrics=['mse'])
    return model


chromaStore = np.load('chroma.npy')
centStore = np.load('spectral-centroid.npy.npy')
rolloffStore = np.load('spectral-rolloff.npy')
mfccsStore = np.load('mel-cepstral-coeffs.npy')
timbreInput = np.vstack(np.load('timbre.npy'))
timbre = timbreInput

tempoStore = []

# ground truth values -> can get original annotations from annotations folder (in this case, we use the static annotations
Y = np.load('/Extracted-Features/static-standardized-valence.npy')

model = baseline_model()
model.fit([chromaStore[10:], mfccsStore[10:], rolloffStore[10:], centStore[10:], timbre[10:]], Y[10:], steps_per_epoch=2000, epochs=5, validation_split = 0.2, validation_steps=1, verbose=True)
print(model.predict([chromaStore[:10], mfccsStore[:10], rolloffStore[:10], centStore[:10], timbre[:10]]))

'''
# save the model, if needed

model_json = model.to_json()

with open("model-my-nn-fixed.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model-my-nn-fixed.h5")
print("Saved model to disk")

'''
