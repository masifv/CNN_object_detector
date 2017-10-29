import time
import cv2
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.applications.resnet50 import preprocess_input
import xml.etree.ElementTree as et

# module level variables
IMAGES_FOLDER = "/home/masi/Projects/CNN_object_localizer/VOCdevkit/VOC2012/JPEGImages"
META_DATA_FOLDER = "/home/masi/Projects/CNN_object_localizer/VOCdevkit/VOC2012/Annotations"

INPUT_SHAPE = (540, 1024, 3)

PREPROCESSED_DATA_DIR = "../preprocessed_data"
TRAIN_DATA_FILE = PREPROCESSED_DATA_DIR + os.sep + "train_data"
VALIDATION_DATA_FILE = PREPROCESSED_DATA_DIR + os.sep + "validation_data"
TEST_DATA_FILE = PREPROCESSED_DATA_DIR + os.sep + "test_data"

def get_data(images_folder: str, meta_data_folder: str):
    """Parse data from PASCAL VOC xml files"""

    print("Parsing data..", end='')
    start = time.time()

    meta_filenames = [os.path.join(meta_data_folder, fn) for fn in
                      os.listdir(meta_data_folder)]
    data = []
    for i, meta_fn in enumerate(meta_filenames):
        root = et.parse(meta_fn).getroot()

        fn_object = root.find('filename')
        filename = fn_object.text

        w = int(root.find('size').find('width').text)
        h = int(root.find('size').find('height').text)
        d = int(root.find('size').find('depth').text)
        shape = (w, h, d)

        for o in root.iter("object"):
            name = o.find('name').text

            # get annotations with object element
            xmin = float(o.find('bndbox').find('xmin').text)
            xmax = float(o.find('bndbox').find('xmax').text)
            ymin = float(o.find('bndbox').find('ymin').text)
            ymax = float(o.find('bndbox').find('ymax').text)

            data.append(
                [filename, meta_fn, name, shape, xmin, xmax, ymin, ymax])

    df = pd.DataFrame(data, columns=['filename', 'meta_filename', 'name',
                                     'shape', 'xmin', 'xmax', 'ymin', 'ymax'])
    df['filename'] = df['filename'].apply(
        lambda x: os.path.join(images_folder, x))

    print("...done in {:0.2f} seconds".format(time.time() - start))
    return df


def create_random_window(image_size):
    """Create a random window within """
    xmin, xmax = np.sort(np.random.randint(0, image_size[1], 2))
    ymin, ymax = np.sort(np.random.randint(0, image_size[0], 2))
    return xmin, xmax, ymin, ymax


def compute_window_overlap(win1, win2):
    """Compute overlap between two windows"""
    xset = set(np.arange(win1[0], win1[1]))
    yset = set(np.arange(win1[2], win1[3]))
    xset2 = set(np.arange(win2[0], win2[1]))
    yset2 = set(np.arange(win2[2], win2[3]))
    xint = xset.intersection(xset2)
    yint = yset.intersection(yset2)
    xuni = xset.union(xset2)
    yuni = yset.union(yset2)
    return (len(xint) / len(xuni)) * (len(yint) / len(yuni))


def generate_negative_samples(df: pd.DataFrame):
    """Generate samples for background class from classes other than 'person'"""
    start = time.time()
    negatives = []
    df_neg = df[~df['name'].str.contains('person')]
    for ind, row in df_neg.iterrows():
        # row = row.copy()
        row['name'] = 'background'
        row['xmin'] = 0
        row['xmax'] = row['shape'][0]
        row['ymin'] = 0
        row['ymax'] = row['shape'][1]
        negatives.append(row)
    print(
        "Sample generation took: {:0.2f} seconds".format(time.time() - start))
    return negatives


def prepare_dataset(images_folder: str, meta_data_folder: str,
                    save: bool = False, load: bool = False):
    """Prepare dataset for training
    Parse xml files, create background class, remove samples with multiple
    persons (because this is localization, not object detection),
    randomize data, split data to training, validation and test partitions
    (50-25-25). Allow saving and previously prepared dataset from file"""

    if load and os.path.exists(TRAIN_DATA_FILE) and os.path.exists(
            VALIDATION_DATA_FILE) and os.path.exists(TEST_DATA_FILE):
        print('loading dataset..', end='')
        train_data = pd.read_pickle(TRAIN_DATA_FILE)
        validation_data = pd.read_pickle(VALIDATION_DATA_FILE)
        test_data = pd.read_pickle(TEST_DATA_FILE)
        print('..done')
    else:
        data = get_data(images_folder, meta_data_folder)
        # generate negative samples
        negatives = generate_negative_samples(data)
        data = data.append(negatives)
        # select only rows with persons or generated negative samples (background)
        data = data[
            (data['name'] == 'person') | (data['name'] == 'background')]
        # remove images with more than one person or background
        data.drop_duplicates(['filename'], keep=False, inplace=True)
        # randomize
        data = data.sample(frac=1).reset_index()
        # split
        split = int(data.shape[0] / 2)
        train_data = data.iloc[1:split]
        test_data = data.iloc[split:]
        validation_data = test_data.sample(frac=0.5, replace=False)
        test_data = test_data[~test_data.index.isin(validation_data.index)]

        print("Data splits: "
              "\n     train data: {:0.2f}% ({})"
              "\n     validation data: {:0.2f}% ({})"
              "\n     test data: {:0.2f}% ({})"
              .format(100 * train_data.shape[0] / data.shape[0],
                      train_data.shape[0],
                      100 * validation_data.shape[0] / data.shape[0],
                      validation_data.shape[0],
                      100 * test_data.shape[0] / data.shape[0],
                      test_data.shape[0]))

        if save:
            print('saving dataset..', end='')
            pd.to_pickle(train_data, TRAIN_DATA_FILE)
            pd.to_pickle(validation_data, VALIDATION_DATA_FILE)
            pd.to_pickle(test_data, TEST_DATA_FILE)
            print('..done')

    return train_data, validation_data, test_data


# module level variables
data = prepare_dataset(IMAGES_FOLDER, META_DATA_FOLDER, load=True, save=False)
train_data, validation_data, test_data = data
