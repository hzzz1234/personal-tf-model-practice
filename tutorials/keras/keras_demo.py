from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers


# #VGG16
# conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150, 150, 3))
# conv_base.summary()
#
# base_dir = '/home/hadoop-aipnlp/cephfs/data/zhen.huaz/cnn/cats_and_dogs_small'
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')
#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.summary()
#
# train_datagen = ImageDataGenerator( rescale=1./255, rotation_range=40,
#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                     target_size=(150, 150), batch_size=20, class_mode='binary')
# validation_generator = test_datagen.flow_from_directory( validation_dir,
#                                                         target_size=(150, 150), batch_size=20, class_mode='binary')
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
# history = model.fit_generator( train_generator, steps_per_epoch=100,
#                               epochs=50, validation_data=validation_generator, validation_steps=50,workers=5)

# from keras.applications.vgg16 import preprocess_input, decode_predictions
# from keras.preprocessing import image
# from keras.datasets import imdb
# from keras.layers import Embedding
#
#
# img_path = '/Users/zhen.huaz/work/github/datesets/cats_and_dogs_small/test/cats/cat.1700.jpg'
#
# img = image.load_img(img_path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# import sys
# sys.path.append("../../utils")
# import draweval
# draweval.draw_loss_valloss([1,2],[3,4])

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

a=K.constant(np.array([1,0]))
b=K.constant(np.array([0,1]))

c=keras.metrics.binary_crossentropy(a,b)
sess = tf.Session()
x=sess.run(c)
print(x)
# class ActivationLogger(keras.callbacks.Callback):
#
#     def set_model(self, model):
#         super().set_model(model)
#
#     def set_params(self, params):
#         super().set_params(params)
#
#     def on_epoch_begin(self, epoch, logs=None):
#         super().on_epoch_begin(epoch, logs)
#
#     def on_epoch_end(self, epoch, logs=None):
#         super().on_epoch_end(epoch, logs)
#
#     def on_batch_begin(self, batch, logs=None):
#         super().on_batch_begin(batch, logs)
#
#     def on_batch_end(self, batch, logs=None):
#         super().on_batch_end(batch, logs)
#
#     def on_train_begin(self, logs=None):
#         super().on_train_begin(logs)
#
#     def on_train_end(self, logs=None):
#         super().on_train_end(logs)