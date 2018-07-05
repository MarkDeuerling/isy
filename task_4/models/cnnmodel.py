from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class CNNModel:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return CNNModel.img_rows, CNNModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding="same"))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))  # kernel required?
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
