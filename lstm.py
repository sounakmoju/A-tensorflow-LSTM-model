import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.layers import Dense,Dropout,LSTM
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.losses import sparse_categorical_crossentropy

from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255
x_test=x_test/255

model=Sequential
model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
opt=Adam(lr=0.001,decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))
          
          
