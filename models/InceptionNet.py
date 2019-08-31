from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2


def InceptionNet(x_train, y_train, params):
    base_model = InceptionResNetV2(
        include_top=False,
        input_shape=(params["img_size"], params["img_size"], 3),
        weights="imagenet",
    )
    for layer in base_model.layers[: params["freeze_layers"]]:
        layer.trainable = False
    for layer in base_model.layers[params["freeze_layers"] :]:
        layer.trainable = True

    # for layer in base_model.layers:
    #     print(layer, layer.trainable)

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    model.add(Dropout(0.5))
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
