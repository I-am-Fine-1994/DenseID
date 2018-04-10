from math import ceil
import numpy as np
from lfw_reader import lfw_reader

total_classes = 5749
total_files = 13233
lfw = lfw_reader()

# preshuffle
index = np.arange(total_files)
np.random.seed(1024)
np.random.shuffle(index)

# split validation
split_rate = 0.25
val_idx = ceil(split_rate * total_files)
train_idx = total_files - val_idx

val_file = index[-val_idx:]
train_file = index[:train_idx]

epochs = 20
batch_size = 128
steps_per_epoch = train_idx//batch_size

from denseid import DenseID
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.optimizers import SGD, Adam, Nadam, RMSprop

model = DenseID()
opt = RMSprop()
model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

for _ in range(epochs):
    print("Epoch: %d/%d" % ((_+1), epochs))
    for step in range(steps_per_epoch):
        # train
        idx = train_file[step*batch_size:(step+1)*batch_size]
        data_batch, label_batch = lfw.load_lfwcrop_data_batch(idx)
        label_batch = to_categorical(label_batch, total_classes)
        hist = model.train_on_batch(data_batch, label_batch)
        # val
        val_choice = np.random.choice(val_idx, batch_size)
        idx_ = val_file[val_choice]
        val_data_batch, val_label_batch = lfw.load_lfwcrop_data_batch(idx_)
        val_label_batch = to_categorical(val_label_batch, total_classes)
        val_hist = model.test_on_batch(val_data_batch, val_label_batch)


        print("Epoch: %d, Step: %d, loss: %f, tr acc: %0.2f%%, val loss: %f, val acc: %0.2f%%" %
               ((_+1), (steps_per_epoch*_)+step+1, hist[0], hist[1]*100, val_hist[0], val_hist[1]*100))

    np.random.shuffle(train_file)
    if ((_+1)%10) == 0:
        model.save("./model_save/denseid_model_"+str(_+1).zfill(4)+".h5")
