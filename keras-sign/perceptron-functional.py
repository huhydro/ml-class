# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape, Add, Concatenate, Input
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

# you may want to normalize the data here..

# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# create model
inp = Input(shape=(28,28))
flat_inp = Flatten()(inp)

reshape = Reshape((img_width,img_height,1),input_shape=(img_width,img_height))(inp)
conv2d_1 = Conv2D(8,(2,2),activation="relu",padding='same')(reshape)
maxpool2d_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv2d_1)
conv2d_2 = Conv2D(32,(3,3),activation="relu",padding='valid')(maxpool2d_1)
maxpool2d_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv2d_2)
conv2d_3 = Conv2D(64,(3,3),activation="relu",padding='valid')(maxpool2d_2)
#maxpool2d_3 = MaxPooling2D(pool_size=(7, 7), strides=None, padding='valid', data_format=None)(conv2d_3)

flat_out = Flatten()(conv2d_3)
flat_sum = Concatenate()([flat_inp,flat_out])

dense1_out = Dense(800, activation="relu")(flat_sum)
dropout_1  = Dropout(0.2)(dense1_out)
dense2_out = Dense(200, activation="relu")(dropout_1)
dropout_2  = Dropout(0.2)(dense2_out)

dense_sum = Concatenate()([dropout_1, dropout_2])

dense_last_out = Dense(num_classes, activation="softmax")(dense_sum)
model = Model(inp, dense_last_out)


model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])
model.summary()
# Fit the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
