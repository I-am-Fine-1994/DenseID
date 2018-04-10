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
label = onehot_label[index]

# split validation
split_rate = 0.1
val_idx = ceil(split_rate * 13233)
val_data = data[:val_idx]
val_label = onehot_label[:val_idx]

train_data = data[val_idx:]
train_label = onehot_label[val_idx:]
print(train_data.shape)
print(train_label.shape)

from denseid import DenseID

epochs = 1
batch_size = 256
steps_per_epch = train_data.shape[0]//batch_size

model = DenseID()
opt = Nadam()
model.compile(optimizer=opt, loss="categorical_crossentropy")

for _ in range(epochs):
    print("epochs: %d/%d" % _, epochs)
    for steps in range(steps_per_epch):
        print("    step=%d" % steps)
        data_batch = train_data[steps*batch_size:(steps+1)*batch_size]
        label_batch = train_label[steps*batch_size:(steps+1)*batch_size]
        model.train_on_batch(data_batch, label_batch)
    train_data = train_data[index]
    train_label = train_label[index]
    model.save("denseid_model_"+ str(_)+".h5")
