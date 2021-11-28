import os
import shutil
import pdb



DATASET_PATH = 'dataset_directory/'



def equalizing_number_of_images_per_class(count_per_class):
  # looping through each class folder one by one.
  for class_name in os.listdir(DATASET_PATH):
    images_per_class = os.listdir(DATASET_PATH + class_name)
    number_of_images_to_delete = len(images_per_class) - count_per_class

    # deleting specific number of images from current class folder.
    for i in range(number_of_images_to_delete):
      os.remove(DATASET_PATH + class_name + '/' + images_per_class[-1])
      images_per_class.pop()






equalizing_number_of_images_per_class(1000)

