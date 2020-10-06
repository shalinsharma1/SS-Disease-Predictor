# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:21:07 2020

@author: Shalin
"""


import numpy as np # linear algebra
import pandas as pd
import os

import cv2
from PIL import Image
import matplotlib.pyplot as plt

################ PARAMETERS ########################            
path = 'C:/Users/Shalin/.spyder-py3/chest_xray'
testRatio = 0.2
valRatio = 0.2
imageDimensions= (32,32,3)
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 2000  
####################################################
 
#### IMPORTING DATA/IMAGES FROM FOLDERS 
count = 0

myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for dirname, _, filenames in os.walk('chest_xray'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_norm = "C:/Users/Shalin/.spyder-py3/chest_xray/train/NORMAL"
train_pneu = "C:/Users/Shalin/.spyder-py3/chest_xray/test/PNEUMONIA"
val_norm = "C:/Users/Shalin/.spyder-py3/chest_xray/val/NORMAL"
val_pneu = "C:/Users/Shalin/.spyder-py3/chest_xray/val/PNEUMONIA"
test_norm = "C:/Users/Shalin/.spyder-py3/chest_xray/test/NORMAL"
test_pneu = "C:/Users/Shalin/.spyder-py3/chest_xray/test/PNEUMONIA"



infected_images = []
for file in os.listdir(train_pneu):
    img = Image.open(os.path.join(train_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    infected_images.append(img)
for file in os.listdir(val_pneu):
    img = Image.open(os.path.join(val_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    infected_images.append(img)
print(len(infected_images))


normal_images = []
for file in os.listdir(train_norm):
    img = Image.open(os.path.join(train_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    normal_images.append(img)
for file in os.listdir(val_norm):
    img = Image.open(os.path.join(val_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    normal_images.append(img)
print(len(normal_images))
X_train = np.asarray(infected_images + normal_images)
y_train = np.asarray([1 for _ in range(len(infected_images))] + [0 for _ in range(len(normal_images))])
print(X_train.shape)
print(y_train.shape)


X_train = X_train.reshape((1747, 36, 36,1))
print(X_train.shape)
print(y_train.shape)
test_infected_images = []
for file in os.listdir(test_pneu):
    img = Image.open(os.path.join(test_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    test_infected_images.append(img)
test_normal_images = []
for file in os.listdir(test_norm):
    img = Image.open(os.path.join(test_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    test_normal_images.append(img)
X_test = np.asarray(test_infected_images + test_normal_images)
y_test = np.asarray([1 for _ in range(len(test_infected_images))] + [0 for _ in range(len(test_normal_images))])

print(X_test.shape)
print(y_test.shape)

X_test = X_test.reshape((624, 36, 36,1))
print(X_test.shape)
print(y_test.shape)
X = np.asarray(infected_images + test_infected_images + normal_images + test_normal_images)
y = np.asarray([1 for _ in range(len(infected_images)+len(test_infected_images))] + [0 for _ in range(len(test_normal_images)+len(normal_images))])
print(X.shape)
print(y.shape)
X = X.reshape((5856, 36, 36, 1))
print(X.shape)
print(y.shape)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes = 2)

from sklearn.utils import shuffle
X, y = shuffle(X, y)
X = X / 255.0
for i in range(10):
    print(y[i])
    plt.imshow(X[i].reshape((36, 36)))
    plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape = X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose = 1)
pred = [np.argmax(i) for i in model.predict(X_test)]
pred[:5]
tru = [np.argmax(i) for i in y_test]
from sklearn.metrics import confusion_matrix
confusion_matrix(tru, pred)
model.save('pneumonia2.h5')
