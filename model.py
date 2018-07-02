import csv
import cv2
import numpy as np
import os
import pickle

from keras.backend import tf as ktf
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Conv2D, Cropping2D, Dropout, Dense, Flatten, Input, Lambda, MaxPooling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import resize

# setup and hyperparameters
batch_size = 128
driving_data_fname = '../driving_data/final_data/'
driving_log_fname = 'driving_log.csv'
driving_img_fname = 'IMG/'
num_cameras = 3
steering_correction_value = .2
conv_dropout = .1
dropout = .4
images_folder = os.path.join(driving_data_fname, driving_img_fname)

# generator function to serve batches
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while True:
        for start in range(0, num_samples, batch_size):
            batch_samples = samples[start:start+batch_size]
            images = []
            measurements = []

            for sample in batch_samples:
                path_name = sample[0]
                measurement = sample[1]
                tmp_img = cv2.imread(path_name)
                tmp_img = tmp_img[..., ::-1]  # turn into rgb
                images.append(tmp_img)
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

# pre-process driving log and add additional steering values for 
# off-center cameras
print ('Generating data structure ...')
with open(os.path.join(driving_data_fname, driving_log_fname)) as csv_file:
    reader = csv.reader(csv_file)

    image_paths = []
    measurements = []

    for line in reader:
        if line[0] == 'center':
            continue  # skip first line if header
        for i in range(num_cameras):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            cur_path = os.path.join(images_folder, filename)
            tmp_measurement = float(line[3])
            if 'left' in filename:
                tmp_measurement += steering_correction_value
            elif 'right' in filename:
                tmp_measurement -= steering_correction_value
            image_paths.append(cur_path)
            measurements.append(tmp_measurement)
    XY = list(zip(image_paths, measurements))


# create train and dev sets through train_test_split
# automatically shuffles the data as well
print ('Splitting data ...')
train_XX, validation_XX = train_test_split(XY, test_size=.2)
train_generator = generator(train_XX)
valid_generator = generator(validation_XX)

# Construct and train model
print ('Building model ...')
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# normalize data
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))

# convolutional blocks
model.add(Conv2D(24, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Conv2D(64, 3, 3, subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Conv2D(64, 3, 3, subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))

model.add(Flatten())

# dense blocks
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1))

# add early stopping call-back function
early_stopping = EarlyStopping(monitor='val_loss')

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(
    train_generator, 
    samples_per_epoch=len(train_XX),
    validation_data=valid_generator,
    nb_val_samples=len(validation_XX),
    nb_epoch=25,
    callbacks=[early_stopping])

print (model.summary())

print ('Storing history file ...')
with open('history.pckl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('model.h5')