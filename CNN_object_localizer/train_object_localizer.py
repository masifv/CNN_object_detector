"""
Object localization with ResNet50 architecture. Uses pre-trained features
from imagenet (transfer learning).

Input is a color image. The network has two outputs, object
class ([1, 0] for person and [0, 1] for background) and its bounding box
coordinates (xmin, xmax, ymin, ymax). Class output (two neurons) uses binary
cross-entropy (categorical loss) and bounding box output (4 neurons) uses L2
loss (regression loss). Total loss is a weighted sum of the losses and
currently the weights are hand tuned. It is also possible to learn it.

NOTE TO SELF: check that magnitude of both losses are in balance, this can be
done by computing a weighted sum as the final loss. The weighting can also be a
learned parameter. When learning the parameter, it is good idea to use some
other metric than the loss to validate your model than loss because this
hyperparameter changes the loss function itself.

NOTE TO SELF: Sometimes two outputs with different losses are handled by first
freezing convolution layers and training each of the outputs separately.
Afterwards the whole network is fine tuned.
"""

import os
import random
import time

import cv2
import pandas as pd
from keras import callbacks
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Input, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress some tf warnings
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, np

from CNN_object_localizer import data

TENSORBOARD_LOG_PATH = "../logs"
CHECKPOINT_ROOT_DIR = "../model_checkpoints"


def train_generator(data: pd.DataFrame, batch_size: int,
                    fixed_input_shape: tuple):
    """ generates one batch at a time that is fed to the fit method """

    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros(
        (batch_size, fixed_input_shape[0], fixed_input_shape[1],
         fixed_input_shape[2]))
    batch_labels = np.zeros((batch_size, 2))

    batch_box = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, len(data.index) - 1)
            x = cv2.imread(data["filename"].iloc[index])
            x = cv2.resize(x, (fixed_input_shape[1], fixed_input_shape[0]))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x.astype(np.float64))
            batch_features[i, :, :, :] = x

            # labels: person == [1, 0], background = [0, 1]
            if data['name'].iloc[index] == 'person':
                batch_labels[i, :] = [1, 0]
            elif data['name'].iloc[index] == 'background':
                batch_labels[i, :] = [0, 1]
            else:
                raise ValueError('Unexpected value in "name" column')

            batch_box[i, 0] = data["xmin"].iloc[index]
            batch_box[i, 1] = data["xmax"].iloc[index]
            batch_box[i, 2] = data["ymin"].iloc[index]
            batch_box[i, 3] = data["ymax"].iloc[index]

        yield (batch_features, [batch_labels, batch_box])


# generates one batch at a time that is fed to the fit method
def validation_generator(data: pd.DataFrame, batch_size: int,
                         fixed_input_shape: tuple):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, fixed_input_shape[0],
                               fixed_input_shape[1], fixed_input_shape[2]))
    batch_labels = np.zeros((batch_size, 2))

    # [xmin, xmax, ymin, ymax]
    batch_box = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, len(data.index) - 1)
            x = cv2.imread(data["filename"].iloc[index])
            x = cv2.resize(x, (fixed_input_shape[1], fixed_input_shape[0]))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x.astype(np.float64))
            batch_features[i, :, :, :] = x

            # labels: person == [1, 0], background = [0, 1]
            if data['name'].iloc[index] == 'person':
                batch_labels[i, :] = [1, 0]
            elif data['name'].iloc[index] == 'background':
                batch_labels[i, :] = [0, 1]
            else:
                raise ValueError('Unexpected value in "name" column')

            batch_box[i, 0] = data["xmin"].iloc[index]
            batch_box[i, 1] = data["xmax"].iloc[index]
            batch_box[i, 2] = data["ymin"].iloc[index]
            batch_box[i, 3] = data["ymax"].iloc[index]

        yield (batch_features, [batch_labels, batch_box])


def train():
    """Construct model and train it"""

    inputs = Input(shape=data.INPUT_SHAPE)

    # load pre-trained model
    base_model = ResNet50(input_shape=data.INPUT_SHAPE, weights='imagenet',
                          include_top=False)

    # freeze pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(inputs)
    # flatten multi-dimensional structure into one dimension
    x = Flatten()(x)

    class_output = Dense(2, activation='softmax')(x)
    # no idea what activation would be good here
    box_coordinates = Dense(4, activation='relu')(x)

    model = Model(inputs=inputs, outputs=[class_output, box_coordinates])
    # compile the model
    # (this should be done after setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss=['binary_crossentropy', 'mean_squared_error'],
                  loss_weights=[1 / 0.6, 1 / 10000])

    # callbacks

    # saves network architecture and weights after each epoch
    cp_dir = CHECKPOINT_ROOT_DIR + os.sep + time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(cp_dir)
    filepath = cp_dir + os.sep + \
               "model_epoch{epoch:02d}_val_loss{val_loss:.2f}.hdf5"
    model_checkpoint = callbacks.ModelCheckpoint(filepath,
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=False,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)
    # writes information about training process to the specified folder in
    # TesorBoard format. TensorBoard can then visualize these logs in a
    # browser
    batch_size = 10
    tensor_board = callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_PATH,
                                         histogram_freq=0,
                                         batch_size=batch_size,
                                         write_graph=True,
                                         write_grads=False,
                                         write_images=False,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None,
                                         embeddings_metadata=None)
    # `steps_per_epoch` is the number of batches to draw from the generator
    # at each epoch.
    model.fit_generator(generator=train_generator(data.train_data,
                                                  batch_size,
                                                  data.INPUT_SHAPE),
                        validation_data=validation_generator(data.validation_data,
                            batch_size,
                            data.INPUT_SHAPE),
                        validation_steps=3,
                        steps_per_epoch=500,
                        epochs=10,
                        workers=1,
                        callbacks=[model_checkpoint, tensor_board])

    return model


if __name__ == '__main__':
    model = train()

    # TODO: Some kind of command line tool to test te network would be nice
    # TODO: plotly train error
