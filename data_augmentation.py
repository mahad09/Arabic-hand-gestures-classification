from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os


DATASET_PATH = 'full_dataset/'
IMAGE_PATH = 'AIN (1).JPG'

datagen = ImageDataGenerator(
  # width_shift_range=0.2
  # height_shift_range=0.2
  # brightness_range=[0.3,1.7]
  zoom_range=[0.8,1.2]
  )

# generate samples and plot
img = load_img(IMAGE_PATH)
data = img_to_array(img)
samples = expand_dims(data, 0)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
  # define subplot
  plt.subplot(330 + 1 + i)
  # generate batch of images
  batch = it.next()
  # convert to unsigned integers for viewing
  image = batch[0].astype('uint8')
  # plot raw pixel data
  plt.imshow(image)
# show the figure
plt.show()

# new_folder = 'augmented_dataset'
# if not (os.path.isdir(new_folder)): os.mkdir(new_folder) 
# if not os.listdir(new_folder):
#   [os.mkdir(new_folder+'/'+class_name) for class_name in os.listdir(DATASET_PATH)]

# print('generating 9 augmented images from each original image ...')

# for class_name in tqdm(os.listdir(DATASET_PATH)):
#   image_number = 1
#   for image_name in os.listdir(DATASET_PATH + class_name):    
#     img = load_img(DATASET_PATH + class_name + '/' + image_name)
#     data = img_to_array(img)
#     samples = expand_dims(data, 0)
#     it = datagen.flow(samples, batch_size=1)

#     for i in range(9):
#       batch = it.next()
#       image = batch[0].astype('uint8')
#       cv2.imwrite(new_folder+ '/'+class_name+'/'+class_name+'_'+str(image_number)+'_aug_'+str(i)+'.jpg', image)
#       image_number += 1
