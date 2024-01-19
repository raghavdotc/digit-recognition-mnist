import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

trainingData = np.loadtxt("/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data/pixels_1.txt")

x_train = trainingData[:, :784]
y_train = trainingData[:, 784:]
print("x shape", x_train.shape)
print("y shape", y_train.shape)
print(x_train)
print(y_train)

model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(units=25, activation='relu', name='l1'),
    Dense(units=15, activation='relu', name='l2'),
    Dense(units=10, activation='linear', name='l3')
])
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    x_train, y_train,
    epochs=5
)

print(model.get_layer("l1").get_weights())
print(model.get_layer("l2").get_weights())
print(model.get_layer("l3").get_weights())

testData = np.loadtxt("/Users/raghavendra.c/PycharmProjects/digit-rec-mnist/data/test-pixels.txt")
x_test = testData[:, :784]
print("x test shape", x_test.shape)
prev = []
for i in range(len(x_test)):
    test_picture = x_test[i]
    print(test_picture.shape)
    curr = test_picture.reshape(1, 784)
    eq_count = 0
    for idx in range(len(prev)):
        if curr[0][idx] == prev[0][idx]:
            eq_count = eq_count + 1
    if eq_count == 784:
        print("Duplicate")
    prediction = model.predict(curr)
    prev = curr
    print(f" predicting: \n{prediction}")
    print(f" Largest Prediction index: {np.argmax(prediction)}")
    if i > 10:
        break
