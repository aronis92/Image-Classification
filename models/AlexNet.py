from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


def AlexNet(x_train, y_train, params):
    model = Sequential()
    model.add(
        Conv2D(
            filters=48,
            activation="relu",
            input_shape=(params["img_size"], params["img_size"], 3),
            kernel_size=(3, 3),
            padding="same",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(filters=128, activation="relu", kernel_size=(3, 3), padding="same")
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(filters=190, activation="relu", kernel_size=(3, 3), padding="same")
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=192,
            activation="relu",
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=128,
            activation="relu",
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(2048, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(500, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(8, activation="softmax"))
    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
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
