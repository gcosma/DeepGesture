# -*- coding: utf-8 -*-
'''
Dataset: Peru sign Language
No. of classes: 6
Authors: Ali A. Alani
Date: 30/11/2017
'''
#%%
#Import the required pakeges
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import time
# We require this for Theano lib ONLY. 
from keras import backend as K
K.set_image_dim_ordering('th')
from numpy import *
import numpy as np
import os
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from matplotlib import pyplot as plt
#%%
from keras.regularizers import l2 # L2-regularisation
l2_lambda = 0.0001
#%%
#Define the variables
# input image dimensions
img_rows, img_cols = 128, 96
# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1
# Batch_size to train
batch_size = 32
## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 3 types of gestures ('unknown', 'close', 'open')
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 6
# Number of epochs to train (change it accordingly)
nb_epoch = 100
# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3
#%%
# Read the image and convert into training and testing sets
path1 = 'C:\Users\Lenovo\CNN_KERAS_MNIST\Gesture Recognition\unsaac-Peru\Peru_sign'    #path of folder of images    
path2 = 'C:\Users\Lenovo\CNN_KERAS_MNIST\Gesture Recognition\unsaac-Peru\image'  #path of folder to save images    
listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = im.convert('L')
                #need to do some more processing here           
    gray.save(path2 +'\\' +  file, "jpeg")
plt.imshow(img)

imlist = os.listdir(path2)

im1 = array(Image.open('C:\Users\Lenovo\CNN_KERAS_MNIST\Gesture Recognition\unsaac-Peru\image' + '\\'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('C:\Users\Lenovo\CNN_KERAS_MNIST\Gesture Recognition\unsaac-Peru\image'+ '\\' + im2)).flatten()
              for im2 in imlist],'f')
import numpy
np1=numpy.array(immatrix).astype('uint8')
np_convert=numpy.invert(np1)
immatrix_con = np_convert.astype('float32')

label=np.ones((num_samples,),dtype = int)
label[0:624]=0
label[625:1249]=1
label[1250:1874]=2
label[1875:2499]=3
label[2500:3124]=4
label[3125:3749]=5     
     
data,Label = shuffle(immatrix_con,label, random_state=2)
train_data = [data,Label]

img=immatrix_con[3749].reshape(96,128)
plt.imshow(img)
print (train_data[0].shape)
print (train_data[1].shape)
(X, y) = (train_data[0],train_data[1])
#%%
# normalize inputs from 0-255 to 0-1
X= (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + 0.001)
# STEP 1: split X and y into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=4)

# reshape to be [samples][pixels][width][height]
trainX = trainX.reshape(trainX.shape[0], 1, 128, 96).astype('float32')
testX = testX.reshape(testX.shape[0], 1, 128, 96).astype('float32')

# one hot encode outputs
trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
num_classes = testY.shape[1]
#%%
#Build the CNN model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Conv2D(32, 5, 5, border_mode='valid', 
             input_shape=(1, 128, 96), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3, 3, border_mode='valid',  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))#

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dropout(0.2))#
    model.add(Dense(64, activation='relu'))#

    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
model.summary()
model.get_config()
model.get_weights()
model.output_shape
#%%
start_time = time.clock()
# Fit the model
hist1 = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=10, batch_size=200, verbose=2)

end_time = time.clock()
pretraining_time = (end_time - start_time)
print ('Training took %f minutes' % (pretraining_time / 60.))
#%%
# Final evaluation of the model
loss, accuracy = model.evaluate(testX, testY)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

#%%
from sklearn.metrics import confusion_matrix, classification_report
import itertools
# Final evaluation for each class and compute the confusion_matrix
y_pred = model.predict_classes(testX)
p=model.predict_proba(testX) # to predict probability

target_names = ['A', 'B', 'C', 'D', 'E', 'F']
print (classification_report(np.argmax(testY,axis=1), y_pred,target_names=target_names))
confusion_matrix = confusion_matrix(np.argmax(testY,axis=1), y_pred)
#%%
target_names1 = ['0', '1', '2', '3', '4', '5']
plt.imshow(confusion_matrix, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names1, rotation=45)
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "orange")
plt.tight_layout()
plt.title('Confusion Matrix', fontsize='12')
plt.ylabel('True label', fontsize='12')
plt.xlabel('Predicted label', fontsize='12')
plt.show()
#%%
print(hist1.history.keys())
# summarize history for accuracy
#plt.style.use('seaborn-notebook')
#plt.subplot(2, 1, 1)
plt.plot(hist1.history['acc'], color='red')
plt.plot(hist1.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(['Traning', 'Test'], loc='upper left')
plt.grid(True)
plt.show()

# summarize history for loss
#plt.subplot(2, 1, 2)
plt.style.use('seaborn-notebook')
plt.plot(hist1.history['loss'], color='orange')
#plt.plot(history.history['val_loss'])
plt.title('Model Traning Loss')
plt.ylabel('Traning Error Rate (%)')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True)
plt.show()
#%%

# Save the trained weights
ans = raw_input("Do you want to save the trained weights - y/n ?")
if ans == 'y':
    filename = raw_input("Enter file name - ")
    fname = 'C:\Users\Lenovo\CNN_KERAS_MNIST\Gesture Recognition\unsaac-Peru' + str(filename) + ".hdf5"
    model.save_weights(fname,overwrite=True)
else:
    model.save_weights("newWeight_96.hdf5",overwrite=True)

    # Save model as well
    # model.save("newModel.hdf5")
