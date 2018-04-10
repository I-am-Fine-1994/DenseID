from math import ceil

from denseid import DenseID

import numpy as np
import tensorflow as tf
from keras.optimizers import SGD, Adam, Nadam, RMSprop

from sklearn.preprocessing import OneHotEncoder

from lfw_reader import lfw_reader
total_classes = 5749
total_files = 13233
lfw = lfw_reader()
enc = OneHotEncoder(dtype="numpy.float32")

# preshuffle
index = np.arange(total_files)
np.random.seed(1024)
np.random.shuffle(index)

# split validation
split_rate = 0.1
val_idx = ceil(split_rate * total_files)
train_idx = total_files - val_idx

val_file = index[-val_idx:]
train_file = index[:train_idx]



epochs = 2
nb_epochs = epochs
batch_size = 128
steps_per_epoch = train_idx//batch_size

from denseid import DenseID
from keras.utils.np_utils import to_categorical
from keras import backend as K

def mean_pred(y_pred, y_true):
    return K.mean((K.abs(y_pred-y_true)))

model = DenseID()
opt = RMSprop(lr=0.005)
model.compile(optimizer=opt, loss="categorical_crossentropy")

for _ in range(epochs):
    print("Epoch: %d/%d" % ((_+1), epochs))
    for step in range(steps_per_epoch):
        idx = train_file[step*batch_size:(step+1)*batch_size]
        data_batch, label_batch = lfw.load_lfwcrop_data_batch(idx)
        # print(label_batch)
        # label_batch =np.eye(total_classes)[label_batch]
        label_batch = to_categorical(label_batch, total_classes)
        hist = model.train_on_batch(data_batch, label_batch)
        y_pred = model.predict(data_batch)
        correct_pred =K.equal(K.argmax(y_pred, 1), K.argmax(label_batch, 1))
        acc = K.mean(K.cast(correct_pred, "float32"))
        acc = K.eval(acc)
        # print(K.eval(acc))
        # acc = mean_pred(y_pred, label_batch)
        # print(type(acc))
        # model.evalute(data_batch, label_batch)
        print("Epoch: %d, Step: %d, loss: %f, acc: %0.2f%%" %
               ((_+1), (steps_per_epoch*_)+step+1, hist, (acc*100)))

    np.random.shuffle(train_file)
    if (_%10) == 0:
        model.save("denseid_model_"+str(_)+".h5")
