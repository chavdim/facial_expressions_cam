#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from scipy import ndimage # for zooming image
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.preprocessing import normalize

###########################################################################

# model: classification of 7 facial expressions
# visualisation of areas used for classification by the network
# positively highlighted areas indicate features in support of the given class
# negatively highlighted areas indicate features against a given class

###########################################################################
### Load and process data
img_width =48
img_height =48

trainXfile = "../fer2013/Training/fer2013_px.csv"
trainYfile  = "../fer2013/Training/fer2013_L.csv"
#
testXfile = "../fer2013/PrivateTest/fer2013_PrivateTest_px.csv"
testYfile = "../fer2013/PrivateTest/fer2013_PrivateTest_L.csv"

# load data to numpy
"""
x_train = genfromtxt(trainXfile, delimiter=',')
x_train /= 255
x_train = np.reshape(x_train,(x_train.shape[0],48,48,1))
y_data = genfromtxt(trainYfile, delimiter=',')
y_train = np.reshape(y_data,(x_train.shape[0],1))
y_train = keras.utils.to_categorical(y_train, num_classes)

x_test = genfromtxt(testXfile, delimiter=',')
x_test /= 255
x_test = np.reshape(x_test,(x_test.shape[0],48,48,1))
y_data = genfromtxt(testYfile, delimiter=',')
y_test = np.reshape(y_data,(x_test.shape[0],1))
y_test = keras.utils.to_categorical(y_test, num_classes)
"""
###########################################################################
# Compile model
model = Sequential()
model.add(Conv2D(32, (3, 3),strides=(1,1), padding='same',
                 input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_classes, activation='softmax'))
opt = keras.optimizers.Adagrad()
model_name = "facialexp_model.h5"
model.load_weights(model_name)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

###########################################################################
# Parameters

# use images starting from:
start_indx = 700
# number of images used
show_num = 20
#
#classes that are shown, others are skipped
show_classes = [3]
#class activation maps shown 
show_feature_maps = [0,3,5]
#figure size
fig_width = 10
fig_height = 10

use_policy = "use all"
#use_policy = "correct only" #shows only correctly classsified samples
#use_policy = "wrong only"

cmap ="seismic" #linear highlights)
#cmap ="PiYG"    #high/low values only highlighted

###########################################################################
# no need to be altered
index_to_emotion={0:"angry",1:"contempt",2:"fear",
                  3:"happy",4:"sad",5:"surprise",6:"neutral"}
layer_indx1 = 9 #max pooling before gap layer index
layer_indx2 = 10 #gap layer index
# tensor : input image -> last max pooling layer
maxp2Out = K.function([model.layers[0].input],
                                      [model.layers[layer_indx1].output])
for indx in range(start_indx,start_indx+show_num): 
    if np.argmax(y_test[indx]) not in show_classes:
        continue
    c = model.predict(x_test[indx:indx+1,0:])
    predicted_index = np.argmax(c)
    ground_truth = np.argmax(y_test[indx])
    
    #enforce policy
    if use_policy == "use all":
        pass
    if use_policy == "correct only":
        if np.argmax(c) != np.argmax(y_test[indx]):
            continue
    if use_policy == "wrong only":
        if np.argmax(c) == np.argmax(y_test[indx]):
            continue
    ###########################################################################
    ### Calculate class activation maps
    maxp2_output = maxp2Out([x_test[indx:indx+1,0:]])[0]
    w_out = model.layers[layer_indx2+1].get_weights()[0]
    scaled_map_list = []
    t=0
    for i in show_feature_maps:
        correct_node_weights = w_out[0:,i]
        mlist = []#lazy
        temp_map = np.copy(maxp2_output)
        for ii in range(maxp2_output[0].shape[2]):
            temp_map[0][0:,0:,ii] =  maxp2_output[0][0:,0:,ii] * correct_node_weights[ii] 
            mlist.append(temp_map[0][0:,0:,ii] )
        nn = np.array(mlist)
        scaled_map_list.append(nn)
        map_means.append(np.mean(nn))
        t+=1
    scaled_map_list = np.array(scaled_map_list)
    
    ### calculate and save all cams to all_cams
    ###normalize maps ?
    all_cams=[]
    for i in scaled_map_list:
        nn = np.sum(i,axis=0)
        s = ndimage.interpolation.zoom(nn, (4, 4), order=1)
        s =   s / np.linalg.norm(s)
        all_cams.append(s)
    all_cams = np.array(all_cams) 
    
    ###########################################################################
    ### Plotting
    img = np.reshape(x_test[indx],(img_width,img_height))
    fig, axs = plt.subplots(1, len(show_feature_maps)+1, sharex=False)
    fig.set_size_inches(fig_width,fig_height)
    
    min_maps = np.min(all_cams,axis=(0,1,2))
    max_maps = np.max(all_cams,axis=(0,1,2))
    
    axs[0].set_title("input")
    axs[0].imshow(img,cmap="gray")
    t=0
    for i in show_feature_maps:
        title_color = "black"
        if i == ground_truth:
            title_color ="green"
        if i == predicted_index:
            if predicted_index != ground_truth:
                title_color = "red"

        axs[t+1].set_title(index_to_emotion[i]+" cam",   color=title_color)
        axs[t+1].imshow(img,cmap="gray")
        s=all_cams[t]
        axs[t+1].imshow(s,alpha=0.4,cmap=cmap,vmin=-max_maps, vmax=max_maps)
        t+=1
    plt.show()
    ###########################################################################
    ### Log
    print("classified as: ", index_to_emotion[predicted_index])
    print("ground truth: ", index_to_emotion[ground_truth])

