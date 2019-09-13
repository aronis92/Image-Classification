import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


def VGG(x_train, y_train, params):

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.summary()

    # sgd = SGD(lr=0.1, decay=1e-6, momentum =0.9, nesterov=True)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    # es = EarlyStopping(mode="min", verbose=1, patience=10)
    mc = ModelCheckpoint("model.h5", monitor="val_acc", save_best_only=True)
    out = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=1,
        shuffle=True,
        callbacks=[mc],
    )

    return out, model

