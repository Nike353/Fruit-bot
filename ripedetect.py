from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')

import os 
print(os.getcwd()) ##Previous Directory
os.chdir(r"C:\Users\varun\desktop\dl\banana_ripeness_detection\data\train\overripe") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break

#repeating the same to ripe and unripe

os.chdir(r"C:\Users\abhay\desktop\dl\banana_ripeness_detection\data\train\ripe") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break
os.chdir(r"C:\Users\abhay\desktop\dl\banana_ripeness_detection\data\train\unripe") ##Change with your current working directory
print(os.getcwd())  ##Current Working Directory

for path in os.listdir():
    img = load_img(f"{path}")
    x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=".", save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 10:     ## creates 10 image form 1 image 
            break

from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np


### Defining Image size
IMAGE_SIZE = [224, 224]### Loading model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)### Freezing layers
for layer in vgg.layers:  
  layer.trainable = False### adding a 3 node final layer for predicion
x = Flatten()(vgg.output)
prediction = Dense(3, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)### Generating Summary
model.summary()

model.compile( loss='categorical_crossentropy',  
               optimizer='adam',  
               metrics=['accuracy'])

#the training set and test set

train_datagen = ImageDataGenerator(rescale = 1./255,                          
                                    shear_range = 0.2,
                                   zoom_range = 0.2,
                                    horizontal_fli= True)
training_set = train_datagen.flow_from_directory('data/train',
                                           target_size = (224, 224),
                                             batch_size = 16,                             
                                        class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('data/validation',
                                         target_size = (224, 224),
                                         batch_size=16,
                                         class_mode = 'categorical')

#training the model

r = model.fit_generator(training_set,  validation_data=test_set,  epochs=25,steps_per_epoch=len(training_set),validation_steps=len(test_set))

model.save("ripeness.h5")#saving the model
