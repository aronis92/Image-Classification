from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint


def MLP(x_train, y_train, params):
    num_classes = 8
    model = Sequential()
    
    model.add(Dense(256, activation='relu',
                    input_shape = (params["img_size"]*params["img_size"]*3,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
    checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', save_best_only = True)
    
    out = model.fit(
        x_train,
        y_train,
        validation_split = 0.1,
        epochs = params["epochs"],
        batch_size = params["batch_size"],
        verbose = 2,
        shuffle = True,
        callbacks = [checkpoint],
    )

    return out, model
