from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint


def OurNet(x_train, y_train, params):
    model = Sequential()
    
    model.add(Conv2D(filters = 64,
                     activation = 'relu',
                     input_shape = (128, 128, 3),
                     kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 128,
                     activation = 'relu',
                     kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 128,
                     activation = 'relu',
                     kernel_size=(3,3)))
       
    model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=128,
                     activation = 'relu',
                     kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 64,
                     activation = 'relu',
                     kernel_size=(3,3)))
    
    model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 64,
                 activation = 'relu',
                 kernel_size=(3,3)))
    
    model.add(AveragePooling2D(pool_size = (3, 3)))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
#    model.add(Conv2D(filters=32,
#                     activation = 'relu',
#                     input_shape = (128, 128, 3),
#                     kernel_size=(3,3)))#, padding = 'same'))
#    model.add(MaxPooling2D(pool_size = (3, 3)))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation = 'softmax'))
    model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    
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
