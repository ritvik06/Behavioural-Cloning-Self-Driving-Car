import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.layers import Convolution2D,Lambda,Cropping2D
from keras.layers import Activation,BatchNormalization
import numpy as np

def network():
    model = Sequential()

    # Normalization Layer
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(70, 160, 3)))
      
#     # trim image to only see section with road
#     model.add(Cropping2D(cropping=((70,25),(0,0)))) 

    # Convolutional Layer 1
    model.add(Convolution2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Convolution2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 3
    model.add(Convolution2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Convolution2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 5
    model.add(Convolution2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flatten Layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')
    return model

quant_weights = np.load('driving_weights_quant.npy',allow_pickle=True)
#print(quant_weights)

model_new = network()
for i in range(27):
	model_new.layers[i].set_weights(quant_weights[i])

model_new.save('new_model.h5')
