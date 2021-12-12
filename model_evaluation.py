import os
import cv2
import shutil
import random
import pdb
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

REFINED_DATASET_PATH = 'preprocessed_dataset'
BATCH_SIZE = 64
INPUT_SHAPE = (64, 64)


def data_generators():
  print('defining generators of training and validation sets...')
  train_datagen = ImageDataGenerator(validation_split=0.2)

  train_generator = train_datagen.flow_from_directory(
    REFINED_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

  validation_generator = train_datagen.flow_from_directory(
    REFINED_DATASET_PATH, target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

  return train_generator, validation_generator


def model_loading():
  model = load_model('results/EfficientNetB4_results/EfficientNetB4_trained_model.h5')

  return model


def model_evaluation(model, validation_generator, train_generator):

  print(validation_generator.class_indices)

  train_dir = 'augmented_dataset/'
  testing = []
  counter = 0
  actual = []
  for folder in tqdm(os.listdir(train_dir)):
    for image in os.listdir(train_dir + folder + '/'):
      img = cv2.imread(train_dir + folder + '/' + image)
      img = cv2.resize(img, (64, 64))
      img = np.array(img)
      img = img.reshape(1, 64, 64, 3)
      # img = img.flatten()
      img *= 255
      predict = model.predict(img)
      # print(counter, 'actual: ', validation_generator.class_indices[folder],'  ', 'predicted: ', np.argmax(predict, axis=1))
      testing.append(np.argmax(predict))
      actual.append(validation_generator.class_indices[folder])
      counter += 1
    print(counter)

  pdb.set_trace()
    
  print(cnf_matrix(actual, testing))
  print(classification_report(actual, testing))


def cnf_matrix(actual, predict):
  cf_matrix = confusion_matrix(actual, predict)
  fig, ax = plt.subplots(figsize=(20, 20))
  sns.heatmap(cf_matrix, ax=ax, fmt='g')
  plt.show()



train_generator, validation_generator = data_generators()
model = model_loading()
y_pred = model_evaluation(model, validation_generator, train_generator)


