import tensorflow as tf
from keras.models import load_model

model = load_model('./model.h5')
model.get_weights()

