import csv
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.layers import Convolution2D,Lambda,Cropping2D
from keras.layers import Activation,BatchNormalization
import argparse
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split

def network():
    model = Sequential()

    # Normalization Layer
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(100, 800, 3)))
      
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

class TopLevel:
    def __init__(self, model, base_path='', epoch_count=3):
        self.data = []
        self.model = model
        self.epochs = epoch_count  #tune this
        self.correction_factor = 0.2
        self.image_path = './Data/'
        self.driving_log_path = 'info.txt'
        self.training_samples = []
        # self.validation_samples = []
        self.batch_size = 32  #tune this    

    #split train data into training and validation sets

    # def split_data(self,split_ratio):
    #     self.training_samples, self.validation_samples = train_test_split(self.data,test_size= split_ratio)

    #     return None

    #read data from simulator

    def read_data(self):
        with open(self.driving_log_path,'r') as fp:
            for line in fp:
                arr = line.strip().split(" ")
                self.data.append(float(arr[0]))
        return None

    def generator(self, batch_size=32):
        num_samples = len(self.data)
        while 1: # Loop forever so the generator never terminates
            for offset in range(0, num_samples-batch_size, batch_size):
                images = []
                angles = []
                for batch_sample in (offset,offset+batch_size,1):                   
                    name = './Data/' + 'check' + str(batch_sample) + '.png'
                    center_image = cv2.imread(name)
                    center_angle = self.data[batch_sample] #getting the steering angle measurement
                    images.append(center_image)

                    angles.append(center_angle)
                    
                    images.append(cv2.flip(center_image,1))

                    angles.append(center_angle*-1)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    def train_generator(self):
        return self.generator(batch_size=32)

    # def validation_generator(self):
    #     return self.generator(samples=self.validation_samples, batch_size=128)

    def run(self):
        self.model.fit_generator(generator=self.train_generator(), 
                                 epochs = self.epochs,
                                 steps_per_epoch= len(self.data))
        self.model.save('model.h5')

def main():
    parser = argparse.ArgumentParser(description='Train a car to drive itself')
    parser.add_argument(
        '--data-base-path',
        type=str,
        default='./data',
        help='Path to image directory and driving log')

    args = parser.parse_args()

    # Instantiate the pipeline
    pipeline = TopLevel(model=network(), base_path=args.data_base_path, epoch_count=3)
    pipeline.read_data()
#     print(len(pipeline.training_samples))
#     print(len(pipeline.validation_samples))
#     print(len(pipeline.training_samples))
#     print(len(pipeline.validation_samples))
    # pipeline.split_data(split_ratio=0.2)
    pipeline.run()
    
if __name__ == '__main__':   
    main()
    

    


