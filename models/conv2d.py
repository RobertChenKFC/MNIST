import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

train_xs = np.loadtxt('../data/train_xs.out')
train_xs = np.reshape(train_xs, (60000, 28, 28, 1))
train_ys = np.loadtxt('../data/train_ys.out')
test_xs = np.loadtxt('../data/test_xs.out')
test_xs = np.reshape(test_xs, (10000, 28, 28, 1))
test_ys = np.loadtxt('../data/test_ys.out')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_xs, train_ys, 
          epochs=10,
          batch_size=32)

results = model.evaluate(test_xs, test_ys, batch_size=32)
print('loss: ', results[0], ' - acc: ', results[1])

model.save('conv2d.h5')