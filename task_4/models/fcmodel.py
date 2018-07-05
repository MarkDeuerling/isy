from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class FCModel:

    @staticmethod
    def load_inputshape():
        return (784,)

    @staticmethod
    def reshape_input_data(x_train, x_test):
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding="same"))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3)))  # kernel required?
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
