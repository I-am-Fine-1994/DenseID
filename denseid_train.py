from math import ceil

from denseid import DenseID

import numpy as np
import tensorflow as tf
from keras.optimizers import SGD, Adam, Nadam

from sklearn.preprocessing import OneHotEncoder

from lfw_reader import lfw_reader
total_classes = 5749
lfw = lfw_reader()
data, label = lfw.load_lfwcrop_data()
enc = OneHotEncoder()
enc.fit(label)
onehot_label = enc.transform(label).toarray()

# preshuffle
index = np.arange(data.shape[0])
np.random.seed(1024)
np.random.shuffle(index)
data = data[index]
onehot_label = label[index]

# split validation
split_rate = 0.1
val_idx = ceil(split_rate * 13233)
val_data = data[:val_idx]
val_label = onehot_label[:val_idx]

train_data = data[val_idx:]
train_label = onehot_label[val_idx:]

def gen_train_data(train_data, train_label, batch_size=256):
    while True:
        data_batch = np.zeros([batch_size, 64, 64, 3])
        onehot_label_batch = np.zeros([batch_size, 5749])
        cnt = 0
        for i in range(val_idx):
            data_batch[cnt] = train_data[i]
            onehot_label_batch[cnt] = train_label[i]
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield data_batch, onehot_label_batch
                data_batch = np.zeros([batch_size, 64, 64, 3])
                onehot_label_batch = np.zeros([batch_size, 5749])

model = DenseID(classes=total_classes)

batch_size = 128
steps_per_epoch = 13233//batch_size
epochs = 2

opt = Nadam()
model.compile(optimizer=opt, loss="categorical_crossentropy")
model.fit_generator(gen_train_data(train_data, train_label, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=2)

model.save("denseid_model_"+ str(epchs)+".h5")
