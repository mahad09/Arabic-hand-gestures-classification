import os
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping 
import matplotlib.pyplot as plt


NUMBER_OF_IMAGES_PER_CLASS = 5000
DATASET_PATH = 'augmented_dataset'
REFINED_DATASET_PATH = 'preprocessed_dataset' 
BATCH_SIZE = 128
NUMBER_OF_CLASSES = 32
INPUT_SHAPE = (64, 64)
EPOCHS=500


def equalizing_number_of_images_per_class():
  # looping through each class folder one by one.
  if not (os.path.isdir(REFINED_DATASET_PATH)): os.mkdir(REFINED_DATASET_PATH)
  if not os.listdir(REFINED_DATASET_PATH):
    [os.mkdir(os.path.join(REFINED_DATASET_PATH, class_name)) for class_name in os.listdir(DATASET_PATH)]

    print('equalizing_number_of_images_per_class')
    for class_name in tqdm(os.listdir(DATASET_PATH)):
      images_per_class = os.listdir(DATASET_PATH + class_name)

      # moving specific number of images from current class folder to new folder.
      for i in range(NUMBER_OF_IMAGES_PER_CLASS):
        shutil.move(os.path.join(DATASET_PATH,class_name,images_per_class[-1],REFINED_DATASET_PATH,class_name,images_per_class[-1]))
        images_per_class.pop()
        random.shuffle(images_per_class)


def data_normalization():
    print('normalizing images...')
    for class_name in tqdm(os.listdir(REFINED_DATASET_PATH)):
      for image_name in os.listdir(os.path.join(REFINED_DATASET_PATH, class_name)):
        original_image = cv2.imread(os.path.join(REFINED_DATASET_PATH, class_name, image_name))
        original_image *= 255
        cv2.imwrite(os.path.join(REFINED_DATASET_PATH, class_name, image_name), original_image)


def data_generators():
  print('defining generators of training and validation sets...')
  train_datagen = ImageDataGenerator(validation_split=0.2)

  train_generator = train_datagen.flow_from_directory(
    REFINED_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

  validation_generator = train_datagen.flow_from_directory(
    REFINED_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

  return train_generator, validation_generator


def model_architecture_compilation():
  print('model compilation...')
  base_model = EfficientNetB4(weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  for layer in base_model.layers:
    layer.trainable = False

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model


def training_callbacks():
  checkpoints_folder = 'checkpoints'
  if not (os.path.isdir(checkpoints_folder)): os.mkdir(checkpoints_folder)

  checkpoints = ModelCheckpoint(
    os.path.join(os.getcwd(), checkpoints_folder, "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5)"),
    save_weights_only='True',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_freq='epoch'
    )

  earlystopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=7,
    verbose=1,
    mode='max',
    baseline=None,
    restore_best_weights=False
    )

  csv_logger = CSVLogger('training.log')

  return [checkpoints, earlystopping, csv_logger]


def model_training(train_generator, validation_generator, callbacks):
  print('model training...')
  history = model.fit(
      train_generator,
      steps_per_epoch = train_generator.samples // BATCH_SIZE,
      validation_data = validation_generator, 
      validation_steps = validation_generator.samples // BATCH_SIZE,
      epochs = EPOCHS,
      callbacks=callbacks,
      workers=4)

  model.save('trained_model.h5')

  return history, model


def plot_accuracy(history):
    plt.title("Accuracy Graph")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()


def plot_loss(history):
    plt.title("Loss Graph")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


# data preparation (1st milestone)
equalizing_number_of_images_per_class()
data_normalization()
train_generator, validation_generator = data_generators()

# model training (2nd milestone)
model = model_architecture_compilation()
callbacks = training_callbacks()
history, model = model_training(train_generator, validation_generator, callbacks)

# model evaluation and results (3rd milestone)
plot_accuracy(history)
plot_loss(history)
