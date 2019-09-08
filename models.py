import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation

def lenet(input_shape=(32,32,1),classes=10):
    X = Sequential()
    #1
    X.add(Conv2D(filters = 6,kernel_size = 5,strides = 1,activation = 'tanh',input_shape = (32,32,1)))
    #2
    X.add(MaxPooling2D((2,2),strides=(2,2)))
    #3
    X.add(Conv2D(filters = 16,kernel_size = 5,strides = 1,activation = 'tanh',input_shape = (14,14,6)))
    #4
    X.add(MaxPooling2D((2,2),strides=(2,2)))
    #5
    X.add(Flatten())
    X.add(Dense(120,activation='tanh',name='fc1'))
    X.add(Dense(84,activation='tanh',name='fc2'))
    X.add(Dense(classes,activation='softmax'))
    return X

def alexnet(input_shape=(227,227,3),classes=1000):
    X = Sequential()
    #1
    X.add(Conv2D(filters = 96,kernel_size = 11,strides = 4,activation = 'relu',input_shape = (227,227,3),padding='valid'))
    #2
    X.add(MaxPooling2D((3,3),strides=(2,2)))
    #3
    X.add(Conv2D(filters = 256,kernel_size = 5,strides = 1,activation = 'relu',padding='same'))
    #4
    X.add(MaxPooling2D((3,3),strides=(2,2)))
    #5
    X.add(Conv2D(filters = 384,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #6
    X.add(Conv2D(filters = 384,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #7
    X.add(Conv2D(filters = 256,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #8
    X.add(MaxPooling2D((3,3),strides=(2,2)))
    #9
    X.add(Flatten())
    X.add(Dense(4096,activation='relu',name='fc1'))
    X.add(Dropout(0.5))
    X.add(Dense(4096,activation='relu',name='fc2'))
    X.add(Dropout(0.5))
    X.add(Dense(classes,activation='softmax'))
    return X
def alexnet_cifar(input_shape=(32,32,3),classes=10):
    X = Sequential()
    #1
    X.add(Conv2D(filters = 48,kernel_size = 3,strides = 1,activation = 'relu',input_shape = (32,32,3),padding='valid'))
    #2
    X.add(MaxPooling2D((2,2),strides=(2,2)))
    #3
    X.add(Conv2D(filters = 96,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #4
    X.add(MaxPooling2D((2,2),strides=(2,2)))
    #5
    X.add(Conv2D(filters = 192,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #6
    X.add(Conv2D(filters = 192,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #7
    X.add(Conv2D(filters = 256,kernel_size = 3,strides = 1,activation = 'relu',padding='same'))
    #8
    X.add(MaxPooling2D((2,2),strides=(2,2)))
    #9
    X.add(Flatten())
    X.add(Dense(512,activation='relu',name='fc1'))
    X.add(Dropout(0.5))
    X.add(Dense(256,activation='relu',name='fc2'))
    X.add(Dropout(0.5))
    X.add(Dense(classes,activation='softmax'))
    return X
def nin_cifar(input_shape=(32,32,3),classes=10):
    X = Sequential()
    #1
    X.add(Conv2D(filters = 192,kernel_size = 5,strides = 2,activation = 'relu',input_shape = (32,32,3),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #2
    X.add(Conv2D(filters = 160,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #3
    X.add(Conv2D(filters = 96,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #4
    X.add(MaxPooling2D((3,3),strides=(2,2),padding='same'))
    #
    X.add(Dropout(0.5))
    #6
    X.add(Conv2D(filters = 192,kernel_size = 5,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #7
    X.add(Conv2D(filters = 192,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #8
    X.add(Conv2D(filters = 192,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #9
    X.add(MaxPooling2D((3,3),strides=(2,2),padding='same'))
    #5
    X.add(Dropout(0.5))
    #10
    X.add(Conv2D(filters = 192,kernel_size = 3,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #11
    X.add(Conv2D(filters = 192,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #12
    X.add(Conv2D(filters = 10,kernel_size = 1,strides = 1,activation = 'relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
    #13
    X.add(GlobalAveragePooling2D())
    X.add(Activation('softmax'))
    return X