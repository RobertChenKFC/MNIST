import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
import numpy as np

train_xs = np.loadtxt('../data/train_xs.out')
train_ys = np.loadtxt('../data/train_ys.out')
test_xs = np.loadtxt('../data/test_xs.out')
test_ys = np.loadtxt('../data/test_ys.out')

model = Sequential()
model.add(Dense(400, activation='relu', input_dim=784))
model.add(Dropout(0.5))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(train_xs, train_ys, 
          epochs=25,
          batch_size=32)

results = model.evaluate(test_xs, test_ys, batch_size=32)
print('loss: ', results[0], ' - acc: ', results[1])

model.save('flat_nn.h5')