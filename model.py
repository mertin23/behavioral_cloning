import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv
import cv2
import json
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Convolution2D,Conv2D,Lambda, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Reshape, merge, Input, ZeroPadding2D, Activation
from keras.optimizers import Adam

with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = np.array([row for row in reader])
#Because I was keep damaging my training data I decided to split my data(as a backup). The simulator works pretty badly on my laptop, so I might generate some wrong training data.
with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    data3 = np.array([row for row in reader])
with open('driving_log3.csv', 'r') as f:
    reader = csv.reader(f)
    data4 = np.array([row for row in reader])
with open('driving_log6.csv', 'r') as f:
    reader = csv.reader(f)
    data5 = np.array([row for row in reader])
with open('driving_log7.csv', 'r') as f:
    reader = csv.reader(f)
    data6 = np.array([row for row in reader])

#data = data2
data = np.concatenate((data2,data3,data4))
shuffle(data)

#I used all 3 cameras.

images_center = np.array(data[:,0])
image_left = np.array(data[:,1])
image_right = np.array(data[:,2])
images = np.concatenate((images_center,image_left,image_right))
y_data = np.array(data[:,3], dtype=float)
y_data_left = y_data+0.08
y_data_right = y_data-0.08
angles = np.concatenate((y_data, y_data_left, y_data_right))



#I will crop the images based on Nvidia's model

ch, row, col = 3, 66, 200

#20% validation set
X_tr_images, X_val_images, y_tr_angles, y_val_angles = train_test_split(images, angles, test_size=0.2, random_state=0)


def crop_image(img):
    height = img.shape[0]
    width = img.shape[1]
    crop_h = row
    crop_w = col
    y_start = 60
    x_start = int((width - crop_w)/2)
    
    return img[y_start:y_start+crop_h, x_start:x_start+crop_w]


#architecture

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))


model.add(Conv2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
model.add(Conv2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
model.add(Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
model.add(Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))
model.add(Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))
model.add(Flatten())
model.add(Dense(1164, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="elu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="elu"))
                
#generator

def generate_image_batch(images, angles, batch_size = 32):
    total = len(images)
    curr = 0
    while (True):
        image_batch = np.zeros((batch_size,row, col, 3),dtype=float)
        angle = np.zeros((batch_size),dtype=float)
        for i in range(batch_size):
            name = images[curr].strip()
            if (name[:3] == 'IMG'):
                image = mpimg.imread('data/'+name)
            else:
                image = mpimg.imread(name)
            image = crop_image(image)
            image_batch[i] = image
            angle[i] = angles[curr]
            curr = (curr+1)%total
    
        yield shuffle(image_batch, angle)

#31176 samples 5 epochs 32 batch_size
training_data = generate_image_batch(X_tr_images,y_tr_angles,32)
validation_data = generate_image_batch(X_val_images, y_val_angles, 32)

#model.compile(optimizer=Adam(lr=0.0001), loss='mse')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(training_data, samples_per_epoch = len(X_tr_images),
                    validation_data=validation_data,
                    nb_val_samples=len(X_val_images),nb_epoch = 5)

print('generator')
json_string = model.to_json()
model.save_weights('model.h5')
with open('model.json', 'w') as f:
    json.dump(json_string, f)

print ('Model saved')
