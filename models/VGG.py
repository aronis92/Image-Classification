from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


def VGG(x_train, y_train, params):


    input_shape = X_train.shape

    #Instantiate an empty model
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
    ])

    model.summary()

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
