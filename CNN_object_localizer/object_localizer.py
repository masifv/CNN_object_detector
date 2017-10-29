import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress some tf warnings
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from CNN_object_localizer import data


def localize(model):
    """Test model by iterating over all test data rows and visualize class
    scores and bounding boxes"""
    num_images = 1
    batch_features = np.zeros((num_images,
                               data.INPUT_SHAPE[0],
                               data.INPUT_SHAPE[1],
                               data.INPUT_SHAPE[2]))

    for i, row in data.test_data.sample(n=1).iterrows():
        im = cv2.imread(row["filename"])
        x = cv2.resize(im, (data.INPUT_SHAPE[1], data.INPUT_SHAPE[0]))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x.astype(np.float64))
        batch_features[0, :, :, :] = x

        labels, bbox = model.predict(batch_features, batch_size=1)
        labels = labels[0]
        xmin, xmax, ymin, ymax = bbox[0]

        plt.figure()
        plt.imshow(im)
        plt.plot([xmin, xmax, xmax, xmin, xmin],
                 [ymin, ymin, ymax, ymax, ymin], 'r')

        person_percentage = labels[0] / np.sum(labels)
        plt.title("Person: {:0.2f}%".format(100 * person_percentage))
        plt.show(block=True)


if __name__ == '__main__':
    model = load_model('../model_checkpoints/2017-10-27_22-32-19/'
                       'model_epoch04_val_loss10.35.hdf5')
    while True:
        localize(model)
