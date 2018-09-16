# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

print(tf.__version__)



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = \
               fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

print("Train image shape: ", train_images.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

for i in range(5):
    test_picture_name = random.choice(os.listdir("fashion_mnist_test"))
    test_picture = Image.open("fashion_mnist_test\\" + test_picture_name )
    test_picture = np.asarray ( test_picture, dtype="int32")
    test_picture = test_picture / 255.0

    labels = np.load("fashion_mnist_test\\test_labels.npy")

    single_image_result = model.predict(test_picture.reshape(1,28,28))
    print("Original picture: ", class_names[labels[int(test_picture_name[7:-5])]])
    print("Predicted picture class: ",class_names[np.argmax(single_image_result)])


