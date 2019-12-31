# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 00:44:33 2019

@author: rohti
"""

#********************CNN using MNIST data************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#image dimensions
x_train.shape
single_image = x_train[0]
single_image.shape
single_image
plt.imshow(single_image)

#actual labels
y_train

#converting labels to dummy 1
from tensorflow.keras.utils import to_categorical
y_train.shape
y_example = to_categorical(y_train)
y_example.shape
y_example[0]

y_cat_test = to_categorical(y_test,num_classes=10)
y_cat_train = to_categorical(y_train,num_classes=10)

x_train = x_train/255
x_test = x_test/255

scaled_image = x_train[0]

#reshape the data to add color channel - in this case it is 1 as image is Black & white
#batch size, width, height, color channels
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),padding='valid',
                 input_shape=(28,28,1),activation='relu')) #"valid" is for no padding

model.add(MaxPool2D(pool_size=(2,2)))

#falttening out images means taking the image of 28x28 and flatten it out to singular array 784 points
model.add(Flatten())

model.add(Dense(128,activation='relu'))

#output layer - softmax for multi-class
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) # we can add in additional metrics https://keras.io/metrics/
#accuracy across categories

#taining the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',patience=1)
model.fit(x_train, y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stopping])

model.metrics_names

losses = pd.DataFrame(model.history.history)

losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()


print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))

from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x_test)
y_cat_test.shape
predictions.shape

print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)

#predict a new image
my_number = x_test[0]
my_number.shape
plt.imshow(my_number.reshape(28,28))

model.predict_classes(my_number.reshape(1,28,28,1))
