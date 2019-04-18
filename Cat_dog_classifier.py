import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.utils.np_utils import to_categorical
# dimensions of our images.
img_width, img_height = 320, 320# ขนาดของภาพแล้วแต่จะปรับเลยครับ

train_data_dir = 'train' #โฟลเดอร์สำหรับเก็บไฟล์ที่ใช้ train
validation_data_dir = 'valid' # โฟลเดอร์สำหรับเก็บไฟล์ที่ใช้ validation
test_data_dir = 'test' 
# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255) # เราจะรีสเกลให้แคบลงเพื่อที่จะได้เรียนรู้เร็วขึ้น
batch_size = 4
# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory( # ตรงนี้จะสร้างออปเจ็คไว้เทรน
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=4,
        class_mode='binary')

validation_generator = datagen.flow_from_directory( # ตรงนี้จะสร้างออปเจ็คไว้ validation
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=4,
        class_mode='binary')
test_generator = datagen.flow_from_directory( # ตรงนี้จะสร้างออปเจ็คไว้ validation
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=4,
        class_mode='binary')

# ข้างล่างลงไปนี้เป็นการสร้างโมเดลครับ ลองปรับเล่นๆดูได้
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
              optimizer='adadelta',
              metrics=['accuracy'])

nb_epoch = 30 #จำนวน epoch คือรอบที่จะทำให้ Model วิวัฒฯ(ผมไม่รู้ว่าศัพท์ไทยเรียกว่าอะไรครับ)ขึ้นไปเรื่อง
nb_train_samples = 8750 #จำนวน sample ที่นำมาเทรน
nb_validation_samples = 2500 #จำนวน sample ที่นำมา validate

#ตรงนี้ก็คือขั้นตอนการสอนโมเดลจริงๆครับ
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
#สอนเสร็จก็เซฟ
open('model/cnn_all.h5', "wb")
model.save_weights('model/cnn_all.h5')

test_labels = test_generator.classes
test_labels = to_categorical(
        test_labels, num_classes=2)

results = model.evaluate_generator(test_generator,nb_validation_samples)
print('test loss, test acc:', results)
