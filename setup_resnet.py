# https://github.com/shoji9x9/CIFAR-10-By-small-ResNet/blob/master/ResNet-for-CIFAR-10-with-Keras.ipynb

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


cifar = tf.keras.datasets.cifar10

class RESNET_Model:
    def __init__(self):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.model = None

    def build_resnet(self):
        n = 9  # 56 layers
        channels = [16, 32, 64]

        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(channels[0], kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu)(x)

        for c in channels:
            for i in range(n):
                subsampling = i == 0 and c > 16
                strides = (2, 2) if subsampling else (1, 1)
                y = Conv2D(c, kernel_size=(3, 3), padding="same", strides=strides, kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(x)
                y = BatchNormalization()(y)
                y = Activation(tf.nn.relu)(y)
                y = Conv2D(c, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(y)
                y = BatchNormalization()(y)
                if subsampling:
                    x = Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(x)
                x = Add()([x, y])
                x = Activation(tf.nn.relu)(x)

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        outputs = Dense(10, activation=tf.nn.softmax, kernel_initializer="he_normal")(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.type = "resnet" + str(6 * n + 2)

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=5)
        self.model.evaluate(x_test, y_test)

resnet_model = RESNET_Model()
resnet_model.build_resnet()

resnet_model.model.summary()

resnet_model.train()