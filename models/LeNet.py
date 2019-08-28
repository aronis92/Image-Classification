from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping


def LeNet(x_train, y_train, params):
    model = Sequential()
    model.add(
        Conv2D(
            6,
            (3, 3),
            activation="relu",
            input_shape=(params["img_size"], params["img_size"], 3),
        )
    )
    model.add(AveragePooling2D())
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(AveragePooling2D())
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(120, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(84, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(8, activation="softmax"))
    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    es = EarlyStopping(mode="min", verbose=1, patience=20)
    out = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=1,
        shuffle=True,
        callbacks=[es],
    )

    return out, model
