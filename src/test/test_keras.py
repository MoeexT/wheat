#! python3
# -*- encoding: utf-8 -*-
# filename: test_keras.py
# @time: 2018/9/27 13:18

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation

x_train = []
y_train = []
x_test = []
y_test = []

model = Sequential()

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.train_on_batch(x_train, y_train)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)
