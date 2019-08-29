from models.LeNet import LeNet
from models.MLP import MLP
from load_data import load_dataset, show_img
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


images, labels = load_dataset()
print("Images shape = ", images.shape, "\nLabels shape = ", labels.shape)

num_classes = np.unique(labels).size
print("Classes:", num_classes)
images = images.astype(np.float32)
labels = labels.astype(np.int8)
images /= 255

# Reshape data for MLP ONLY.
# Comment out for convolutional networks.
#images = images.reshape(8000, images.shape[1]*images.shape[2]*images.shape[3])

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(labels)
Y = to_categorical(Y, num_classes)

# Data split
x_train, x_test, y_train, y_test = train_test_split(images, Y, test_size=0.1)

print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("x_test shape = ", x_test.shape)
print("y_test shape = ", y_test.shape)


def plot_history(history, params):
    title = "&".join(["%s=%s" % (key, value) for (key, value) in params.items()])
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title(params["model"])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./results/" + title + "_accuracy.png")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(params["model"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./results/" + title + "_loss.png")
    plt.show()


p = {
    "model": "MLP",  # for title
    "img_size": 128,
    # "num_classes": NUM_CLASSES,
    # "freeze_layers": 0,
    "epochs": 30,
    "batch_size": 50,
}

history, model = MLP(x_train, y_train, p)
#history, model = LeNet(x_train, y_train, p)

plot_history(history, p)

results = model.evaluate(x_test, y_test, batch_size=50)
print("test loss, test acc:", results)
