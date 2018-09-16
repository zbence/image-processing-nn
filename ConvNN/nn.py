import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

print(tf.__version__)

def loadCSV(path):

    data = pd.read_csv(path)
    data = data.values
    labels = data[:,0]
    pictures = data[:,1:]
    pictures = pictures.reshape(-1,28,28,1)

    print("Found at " + path +" " + str(labels.shape) +" labels and " + str(pictures.shape) + "images")
    return pictures, labels
                                
train_images, train_labels = loadCSV("..\\ClassTutorial\\MNIST_fashion_raw\\fashion-mnist_train.csv")
test_images, test_labels = loadCSV("..\\ClassTutorial\\MNIST_fashion_raw\\fashion-mnist_test.csv")

train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1),filters=15, kernel_size=(2,2),activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(3,3),strides=3),
    keras.layers.Conv2D(filters=45, kernel_size=(2,2),activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
    ])



model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

for layer in model.layers:
    print("Output shape at ",layer, ": " ,layer.output_shape)

model.fit(train_images, train_labels,epochs=15)

test_loss, test_acc = model.evaluate(test_images,test_labels)

print('Test accuracy:', test_acc)


                
    
