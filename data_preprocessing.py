import os
import cv2
import shutil
import random
import pdb
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


DATASET_PATH = 'preprocessed_dataset/'
BATCH_SIZE = 64
NUMBER_OF_CLASSES = 32
INPUT_SHAPE = (32, 32)
EPOCHS=500


def equalizing_number_of_images_per_class(count_per_class):
  # looping through each class folder one by one.
  for class_name in os.listdir(DATASET_PATH):
    images_per_class = os.listdir(DATASET_PATH + class_name)
    number_of_images_to_delete = len(images_per_class) - count_per_class

    # deleting specific number of images from current class folder.
    for i in range(number_of_images_to_delete):
      os.remove(DATASET_PATH + class_name + '/' + images_per_class[-1])
      images_per_class.pop()
      random.shuffle(images_per_class)


def data_normalization():
  new_folder = 'preprocessed_dataset'
  if not (os.path.isdir(new_folder)): os.mkdir(new_folder) 
  if not os.listdir(new_folder):
    [os.mkdir(new_folder+'/'+class_name) for class_name in os.listdir(DATASET_PATH)]

    for class_name in tqdm(os.listdir(DATASET_PATH)):
      for image_name in os.listdir(DATASET_PATH + class_name):
        original_image = cv2.imread(DATASET_PATH+class_name+'/'+image_name)
        original_image *= 255
        cv2.imwrite(new_folder+ '/'+class_name+'/'+image_name, original_image)



def keras_preprocessing():
  train_datagen = ImageDataGenerator(validation_split=0.2)

  train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

  validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

  return train_generator, validation_generator


def model_architecture_compilation():
  base_model = VGG16(weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  for layer in base_model.layers:
    layer.trainable = False

  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

  return model


def model_training(train_generator, validation_generator):
  model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = EPOCHS,
    workers=1)

  model.save('trained_model.h5')





number_of_images_per_class = 1000
equalizing_number_of_images_per_class(number_of_images_per_class)
data_normalization()
train_generator, validation_generator = keras_preprocessing()
model = model_architecture_compilation()
trained_model = model_training(train_generator, validation_generator)
