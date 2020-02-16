import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import zipfile
from keras import models
from keras import layers
import os, random
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50, InceptionV3
from tensorflow.keras.layers import Embedding
#
# zip_ref = zipfile.ZipFile('/home/ogai/Downloads/dataset_internet/flowers-recognition.zip' , 'r')
# zip_ref.extractall('unzipped_folder') # unzip directory
# zip_ref.close()
# base_path = 'Dataset/flowers/'
base_path = '/home/ogai/Downloads/dataset_internet/flowers'
# daisy：菊花。 dandelion：蒲公英。 rose：玫瑰。 sunflower：向日葵。 tulip：郁金香。
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#Load file names
fnames = []
for category in categories:
    flower_folder = os.path.join(base_path, category)
    file_names = os.listdir(flower_folder)
    full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
    fnames.append(full_path)

print('length for each category:', [len(f) for f in fnames])
#load image
images = []
for names in fnames:
    one_category_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
    images.append(one_category_images)

print('number of images for each category:', [len(f) for f in images])

#Calculate the minimal shape for all images
for i,imgs in enumerate(images):
    shapes = [img.shape for img in imgs]
    widths = [shape[0] for shape in shapes]
    heights = [shape[1] for shape in shapes]
    print('%d,%d is the min shape for %s' % (np.min(widths), np.min(heights), categories[i]))

#Create a function to convert cv2 BGR format to RGB for showing
def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

#Show sample images for all flowers
plt.figure(figsize=(15,10))
for i, imgs in enumerate(images):
    plt.subplot(2,3,i+1)
    idx = np.random.randint(len(imgs))
    plt.imshow(cvtRGB(imgs[idx]))
    plt.grid(False)
    plt.title(categories[i]+' '+str(idx))
plt.show()
#Resize all the images to 256x256
img_width, img_height = 256, 256
img = images[3][659]
print(img.shape)
resized_img = resize(img, (img_width, img_height, 3))
resized_img2 = cv2.resize(img,(img_width, img_height), interpolation = cv2.INTER_CUBIC)
print(resized_img.shape)
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.title('original image (BGR-channel)')
plt.grid(False)
plt.imshow(img)
plt.subplot(2,2,2)
plt.title('original image (RGB-channel)')
plt.grid(False)
plt.imshow(cvtRGB(img))
plt.subplot(2,2,3)
plt.title('resized by skimage (BGR-channel)')
plt.grid(False)
plt.imshow((resized_img))
plt.subplot(2,2,4)
plt.title('resized by opencv (RGB-channel)')
plt.grid(False)
plt.imshow(cvtRGB(resized_img2))
plt.show()
# Apply resize to all images
resized_images = []
for i,imgs in enumerate(images):
    resized_images.append([cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) for img in imgs])
#Split dataset to 80% of training and 20% of validation
train_images = []
val_images = []
for imgs in resized_images:
    train, test = train_test_split(imgs, train_size=0.8, test_size=0.2)
    train_images.append(train)
    val_images.append(test)

#creat labels
len_train_images = [len(imgs) for imgs in train_images]
print(len_train_images)
print('sum of train images:', np.sum(len_train_images))
train_categories = np.zeros((np.sum(len_train_images)), dtype='uint8')
for i in range(5):
    if i is 0:
        train_categories[:len_train_images[i]] = i
    else:
        train_categories[np.sum(len_train_images[:i]):np.sum(len_train_images[:i + 1])] = i

len_val_images = [len(imgs) for imgs in val_images]
print(len_val_images)
print('sum of val_images:', np.sum(len_val_images))
val_categories = np.zeros((np.sum(len_val_images)), dtype='uint8')
for i in range(5):
    if i is 0:
        val_categories[:len_val_images[i]] = i
    else:
        val_categories[np.sum(len_val_images[:i]):np.sum(len_val_images[:i + 1])] = i
#Convert image data to numpy array
tmp_train_imgs = []
tmp_val_imgs = []
for imgs in train_images:
    tmp_train_imgs += imgs
for imgs in val_images:
    tmp_val_imgs += imgs
train_images = np.array(tmp_train_imgs)
val_images = np.array(tmp_val_imgs)

print('Before converting')
print('train data:', train_images.shape)
print('train labels:', train_categories.shape)
train_data = train_images.astype('float32')
val_data = val_images.astype('float32')
train_labels = np_utils.to_categorical(train_categories, len(categories))
val_labels = np_utils.to_categorical(val_categories, len(categories))
print()
print('After converting')
print('train data:', train_data.shape)
print('train labels:', train_labels.shape)
#Shuffle the dataset (set a seed and randomize both the data and the labels)¶
seed = 100
np.random.seed(seed)
np.random.shuffle(train_data)
np.random.seed(seed)
np.random.shuffle(train_labels)
np.random.seed(seed)
np.random.shuffle(val_data)
np.random.seed(seed)
np.random.shuffle(val_labels)
train_data = train_data[:3400]
train_labels = train_labels[:3400]
val_data = val_data[:860]
val_labels = val_labels[:860]
print('shape of train data:', train_data.shape)
print('shape of train labels:', train_labels.shape)
print('shape of val data:', val_data.shape)
print('shape of val labels:', val_labels.shape)

WEIGHTS_PATH_NO_TOP = '/home/ogai/Downloads/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_model_from_VGG19():
    """
      use VGG19
    """
    model = VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers:
        layer.trainable = False

    # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(len(categories), activation="softmax")(x)

    # creating the final model
    final_model = Model(inputs=model.input, outputs=predictions)

    final_model.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['acc'])  # optimizer=RMSprop(lr=0.001)

    return final_model
model_VGG19 = create_model_from_VGG19()
model_VGG19.summary()


# Parameters
batch_size = 32
epochs1 = 5
epochs2 = 50
epochs3 = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(
    rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow(
    train_data,
    train_labels,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    val_data,
    val_labels,
    batch_size=batch_size
)
start = time.time()

model_VGG19_info = model_VGG19.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_data)/batch_size,   # -> 106 # images 3392 = steps * batch_size = 106 * 32
    epochs=epochs2,
    validation_steps=len(val_data)/batch_size, # -> 26 # images 832 = steps * batch_size = 26 * 32
    validation_data=val_generator,
    verbose=2
)

end = time.time()
duration = end - start
print ('\n model_VGG19 took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs2) )

def return_name(label_arr):
  idx = np.where(label_arr == 1)
  return idx[0][0]


def plot_model_history(model_name, history, epochs):
    print(model_name)
    plt.figure(figsize=(15, 5))

    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
    plt.plot(np.arange(1, len(history['val_acc']) + 1), history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss']) + 1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss']) + 1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='best')

    plt.show()

plot_model_history('model_VGG19', model_VGG19_info.history, epochs2)

def predict_one_image(img, model):
  img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
  img = np.reshape(img, (1, img_width, img_height, 3))
  img = img/255.
  pred = model.predict(img)
  class_num = np.argmax(pred)
  return class_num, np.max(pred)


# idx = 120
# pred, probability = predict_one_image(images[4][idx], model_ResNet50)

test_img = cv2.imread( '/home/ogai/Downloads/dataset_internet/rose1.jpg')
pred, probability = predict_one_image(test_img, model_VGG19)
print('%s %d%%' % (categories[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(cvtRGB(test_img))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()

def predict_val(val_data, model):
  val_input = np.reshape(val_data, (1, img_width, img_height, 3))
  val_input = val_input/255.
  pred = model.predict(val_input)
  class_num = np.argmax(pred)
  return class_num, np.max(pred)


print("Model trained from VGG19")
plt.figure(figsize=(15, 15))
for i in range(9):
    idx = np.random.randint(860)

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(cvtRGB(val_data.astype('uint8')[idx]))
    category_idx = return_name(val_labels[idx])

    pred, prob = predict_val(val_data[idx], model_VGG19)
    plt.title('True: %s || Predict: %s %d%%' % (categories[category_idx], categories[pred], round(prob, 2) * 100))
    plt.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.show()