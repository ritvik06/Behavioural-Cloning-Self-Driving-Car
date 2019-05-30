import tensorflow as tf
from keras.models import load_model
import numpy as np

a = []

model = load_model('./model.h5')
a=model.get_weights()
A = np.array(a)
np.save('weights.npy',A)
