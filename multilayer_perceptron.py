import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical, plot_model
from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from load_data import load_dataset, show_img
from processing import create_dir

images, labels = load_dataset()

train_size = int(len(images)*0.8)
test_size = int(len(images)*0.1)
img_size = images[0].shape

# Split data to train, validation and test.
X_train = images[:train_size]
X_valid = images[train_size:(train_size + test_size)]
X_test = images[(train_size + test_size):]
y_train = labels[:train_size]
y_valid = labels[train_size:(train_size + test_size)]
y_test = labels[(train_size + test_size):]

#Reshape the data for the multi layer perceptron
X_train = X_train.reshape(X_train.shape[0], img_size[0]*img_size[1]*img_size[2]).astype('float32')
X_valid = X_valid.reshape(X_valid.shape[0], img_size[0]*img_size[1]*img_size[2]).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_size[0]*img_size[1]*img_size[2]).astype('float32')

#Normalize the data
X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

#Convert to categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

#Define the number of classes
num_classes = y_test.shape[1] # number of categories

#Define the baseline, a multi layer perceptron
def multi_layer_perceptron():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(img_size[0]*img_size[1]*img_size[2],)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
#    model.add(Dense(128, activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    #plot_model(model, to_file="model.png", show_shapes=True)
    
    #Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model


#Build the model
model = multi_layer_perceptron()

architecture = 'MLP_256_128_64_' # Change that to the architectures name
loc = 'multilayer_perceptron/' # Change this to the name of your neural network e.g AlexNet.
create_dir(loc)

#Fit the model
history = model.fit(X_train, y_train,
                    validation_data = (X_valid, y_valid),
                    epochs = 15,
                    batch_size = 50,
                    verbose = 2, 
                    shuffle = True, 
                    callbacks = [ModelCheckpoint(loc + architecture + '_model.h5',
                                                 monitor='val_acc',
                                                 save_best_only = True)])

# summarize history for acc
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(loc + architecture + 'acc.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(loc + architecture + 'loss.png')
plt.close()

scores = model.evaluate(X_test, y_test, verbose = 2)
print("Accuracy: {} \nError: {}".format(scores[1], 100-scores[1]*100))













