import csv
import cv2
import os
import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle



batch_size = 1
recompute = True
driving_data_fname = '../driving_data/test_data/'
driving_log_fname = 'driving_log.csv'
driving_img_fname = 'IMG/'
num_cameras = 3
steering_correction_value = .2
images_folder = os.path.join(driving_data_fname, driving_img_fname)

# pre-process driving log and add additional steering values for 
# off-center cameras
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

# evaluate
test_generator = generator(XY)
model = load_model('model.h5')
#print (next(test_generator))
losses = model.evaluate_generator(
    generator=test_generator,
    val_samples=len(XY))
print (losses)