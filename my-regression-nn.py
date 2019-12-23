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
    # x = Flatten()(x)
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
    # z = Dropout(0.3)(z)
    z = Dense(40, activation='relu')(z)
    z = Dense(20, activation='relu')(z)

    # fourth branch
    a = Dense(100, activation='relu')(rolloff)
    a = Dense(20, activation='relu')(a)

    # fifth branch
    b = Dense(10, activation='relu')(timbre)
    # b = Dense(20, activation='relu')(b)

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


chromaStore = np.load('/h2/vivek/Documents/chromaStoreZ.npy')
centStore = np.load('/h2/vivek/Documents/centStoreZ.npy')
rolloffStore = np.load('/h2/vivek/Documents/rolloffStoreZ.npy')
mfccsStore = np.load('/h2/vivek/Documents/mfccs.npy')
timbreInput = np.vstack(np.load('/h2/vivek/Documents/audio-features.npy'))

# timbre = []
# for i in timbreInput:
#     timbre.append(np.reshape(i, (1, 10)))
#
# timbre = np.stack(timbre)
timbre = timbreInput

# might be in the WRONG format :(

tempoStore = []

# ground truth values
dataframe = pd.read_csv('/net/cvcfs/storage/spotify-datasets/1000-songs/Annotations/static_annotations.csv')
songs = dataframe[['song_id']]
extraneous = ['song_id', 'mean_arousal', 'std_arousal', 'std_valence', 'mean_valence']
dataframe = dataframe.drop(extraneous, axis=1)
print(dataframe.keys())
Y = dataframe.values
# Y = Y[:100]
Y = Y.squeeze()
np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/static-standardized-valence.npy', Y)

centStore = centStore.reshape(744, 200)
rolloffStore = rolloffStore.reshape(744, 200)

chromaStore = chromaStore.reshape(744, 12, 200, 1)
mfccsStore = mfccsStore.reshape(744, 20, 200, 1)

print("Y:            ", Y.shape)
print("chromaStore:  ", chromaStore.shape)
print("mfccsStore:   ", mfccsStore.shape)
print("rolloffStore: ", rolloffStore.shape)
print("centStore:    ", centStore.shape)
print("timbre: ", timbre.shape)


# np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/spectral-centroid.npy', centStore)
# np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/spectral-rolloff.npy', rolloffStore)
# np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/mel-cepstral-coeffs.npy', mfccsStore)
# np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/chroma.npy', chromaStore)
# np.save('/net/cvcfs/storage/spotify-datasets/1000-songs/Extracted-Features/timbre.npy', timbre)

model = baseline_model()
model.fit([chromaStore[10:], mfccsStore[10:], rolloffStore[10:], centStore[10:], timbre[10:]], Y[10:], steps_per_epoch=2000, epochs=5, validation_split = 0.2, validation_steps=1, verbose=True)
print(model.predict([chromaStore[:10], mfccsStore[:10], rolloffStore[:10], centStore[:10], timbre[:10]]))

'''
# save the model

model_json = model.to_json()

with open("model-my-nn-fixed.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model-my-nn-fixed.h5")
print("Saved model to disk")

'''
