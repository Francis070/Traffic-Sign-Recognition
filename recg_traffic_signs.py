# -*- coding: utf-8 -*-
"""
Created on Thu May 14 03:06:57 2020

@author: Abhijit Mukherjee
"""

!git clone https://bitbucket.org/jadslim/german-traffic-signs.git
!ls german-traffic-signs
import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import pickle  #for handling the pickel file
import pandas as pd
import random

np.random.seed(0)#to read the data sequentially

with open('german-traffic-signs/train.p' , 'rb') as f:
  train_data = pickle.load(f)#unpickle the train.p file

with open('german-traffic-signs/valid.p' , 'rb') as f:
  val_data = pickle.load(f)

with open('german-traffic-signs/test.p' , 'rb') as f:
  test_data = pickle.load(f)
  
print(test_data)

x_train, y_train = train_data['features'],train_data['labels']
x_val, y_val = val_data['features'],val_data['labels']
x_test, y_test = test_data['features'],test_data['labels']

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

#assert statement is like a error raising line which checks whether the statement is valid or not
assert(x_train.shape[0]==y_train.shape[0]), "the no. of images is not equal to the no of labels"
assert(x_val.shape[0]==y_val.shape[0]), "the no. of images is not equal to the no of labels"
assert(x_test.shape[0]==y_test.shape[0]), "the no. of images is not equal to the no of labels"

assert(x_train.shape[1:]==(32,32,3)), "the dim of the image is not 32X32X3"
assert(x_val.shape[1:]==(32,32,3)), "the dim of the image is not 32X32X3"
assert(x_test.shape[1:]==(32,32,3)), "the dim of the image is not 32X32X3"

data = pd.read_csv(r"german-traffic-signs/signnames.csv")
print(data)

num_of_samples = []
cols = 5
num_classes = 43

fig , axs = plt.subplots(nrows = num_classes , ncols = cols, figsize=(15,50))
fig.tight_layout()
for i in range(cols):
  for j, row in data.iterrows():
    x_selected= x_train[y_train==j]#finds out all the data belongs to specific category
    axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected)- 1)), :, :],cmap="gray")#randomly selects 5 images from the above list
    axs[j][i].axis("off")#the values os the axis will not be shown in the subplots
    if i==2:
      axs[j][i].set_title(str(j)+'-'+row['SignName'])
      num_of_samples.append(len(x_selected))
      
print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0 , num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("class number ")
plt.ylabel("number of images")

#displaying the inage in a particular index
import cv2
plt.imshow(x_train[1000])
plt.axis('off')
print(x_train[1000].shape)
print(y_train[1000])

#pre-processing the image
def grayscale(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#reduce the image size and increases the processing speed
  return img
img = grayscale(x_train[1000])
plt.imshow(img,cmap='gray')
plt.axis('off')
print(img.shape)


#histogram - equalization for brightness standardadization

def equalize(img):
  img = cv2.equalizeHist(img)#it  helps to distyribute the light throughout the image
  return img
img= equalize(img)
plt.imshow(img,cmap='gray')
plt.axis('off')
print(img.shape)

def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255
  return img

#we use map function
import numpy as np
x_train = np.array(list(map(preprocessing,x_train)))
x_val = np.array(list(map(preprocessing,x_val)))
x_test = np.array(list(map(preprocessing,x_test)))

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

plt.imshow(x_train[1000],cmap="gray")
plt.axis("off")
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0],32,32,1)
x_val = x_val.reshape(x_val.shape[0],32,32,1)
x_test = x_test.reshape(x_test.shape[0],32,32,1)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

#one hot encoding of the image data
y_train= to_categorical(y_train,43)
y_val= to_categorical(y_val,43)
y_test= to_categorical(y_test,43)

#augmentation = adding something virtual to a data
#Fit Generator:- this is the data augmentation technique to further improve the accuracy 
#and generalization of the model    

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range = 0.1,zoom_range=0.2, shear_range= 0.1, rotation_range = 10)
datagen.fit(x_train)
                             
#calling the data generator to augment images in real time

batches = datagen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)#creates 20 no. of augmented images of a single image

fig,axes = plt.subplots(1,15,figsize =(20,5))
fig.tight_layout()

for i in range(15):
  axes[i].imshow(x_batch[i].reshape(32,32), cmap= "gray")
  axes[i].axis('off')

def lenet_model():
  model = Sequential()
  model.add(Conv2D(60,(5,5),input_shape =(32,32,1), activation='relu'))
  model.add(Conv2D(60,(5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(30,(3,3), activation='relu'))
  model.add(Conv2D(30,(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(500 ,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = lenet_model()
print(model.summary())

h = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 50), steps_per_epoch=2000,epochs = 10, validation_data=(x_val,y_val), shuffle = 1, verbose =1)

scores= model.evaluate(x_test,y_test, verbose=0)
print("CNN error :%.2f%%" % (100-scores[1]*100))

import matplotlib.pyplot as plt
plt.plot(h.history["loss"])#loss vs val_loss graph
plt.plot(h.history["val_loss"])
plt.show()

plt.plot(h.history["acc"])#loss vs val_loss graph
plt.plot(h.history["val_loss"])
plt.xlabel('epochs')
plt.legend(['training_acc','Val_acc'])
plt.show()

score = model.evaluate(x_test,y_test, verbose=1)
print("test score:" ,score[0])
print('test_accuracy:',score[1])

#predict internet images to test the model

import requests
from PIL import Image#lib for image .Pillow image lib
url = "https://us.123rf.com/450wm/bwylezich/bwylezich1602/bwylezich160200086/51784427-german-road-sign-stop.jpg"
r = requests.get(url, stream= True)#streaming the image in terms of bytes(encoding to byte)
img = Image.open(r.raw)#creating the image form bytes(decoding from bytes)
plt.imshow(img,cmap=plt.get_cmap('gray'))

#Preproecess image

img = np.asarray(img)#the img is converted into numpy array to use the cv2 package
img = cv2.resize(img, (32,32))
img = preprocessing(img)
plt.imshow(img, cmap= plt.get_cmap('gray'))
print(img.shape)

#Reshape

img = img.reshape(1,32,32,1)
print("predicted category",model.predict_classes(img))
#Test image

print("predicted sign", data["SignName"][model.predict_classes(img)])






