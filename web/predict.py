import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import easygui

img_width, img_height = 512, 512 #ขนาดภาพเหมือนเดิมครับ

#ตรงนี้ก็บอกว่าเราจะเอา weight ไปใช้กับโมเดลอะไร
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('basic_cnn_20_epochs.h5') #โหลด weight มาใช้ครับ

#ข้างล่างก็มี ใช้ easygui เลือกไฟล์มาแล้วลอง predict ว่าเป็นรูปอะไร แล้วก็ prompt บอกว่าเป็นรูปนั้นครับ
while True:
    msg ="click predict to choose pic."
    title = "Dog or Cat"

    choices = ["Predict", "Quit"]
    choice = easygui.choicebox(msg, title, choices)
    if choice == "Predict":
        filename1 = easygui.fileopenbox()

        imgg = load_img(filename1, target_size=(img_width,img_height))
        predictg = img_to_array(imgg)
        predictiong = model.predict_classes(predictg.reshape((1,img_width, img_height,3)),batch_size=16, verbose=0)
        if predictiong[0][0] == 1: easygui.msgbox("dog")
        else : easygui.msgbox("cat")
    else:
        os._exit()