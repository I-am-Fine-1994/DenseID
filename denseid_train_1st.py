from lfw_reader import lfw_reader
import numpy as np

lfw = lfw_reader()
min_num = 5
path_data, label = lfw.form_data_from_nlt(min_num)
# 每个人抽取 0.6 图片作为训练集，0.2 作为验证集，0.2 作为训练集
people_num = label[-1] + 1
split_rate = 0.6
val_rate = 0.2

train_idx = [i for i in range(len(label)) if (i%min_num) < round(min_num*split_rate)]
# print(train_idx)
val_idx = [i for i in range(len(label)) if  round(min_num*(split_rate)) <= (i%min_num) < round(min_num*(split_rate+val_rate))]
# print(val_idx)
test_idx = [i for i in range(len(label)) if (i%min_num) >= round(min_num*(split_rate+val_rate))]
# print(test_idx)

def slice_by_list(data_lst, idx_list):
    data = []
    for idx in idx_list:
        data.append(data_lst[idx])
    return data

def my_shuffle(data, label):
    idx = np.arange(len(label))
    np.random.seed(1024)
    np.random.shuffle(idx)
    shuffle_data = []
    shuffle_label = []
    for i in idx:
        shuffle_data.append(data[i])
        shuffle_label.append(label[i])
    return shuffle_data, shuffle_label

train_data, train_label = slice_by_list(path_data, train_idx), slice_by_list(label, train_idx)
val_data, val_label = slice_by_list(path_data, train_idx), slice_by_list(label, train_idx)
test_data, test_label = slice_by_list(path_data, train_idx), slice_by_list(label, train_idx)

# pre-shuffle
train_data, train_label = my_shuffle(train_data, train_label)
# print(train_label)
val_data, val_label = my_shuffle(val_data, val_label)
test_data, test_label = my_shuffle(test_data, test_label)

epochs = 800
batch_size = 128
steps_per_epoch = len(train_label)//batch_size

# from denseid import DenseID
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.layers import Dense
from keras.applications import densenet

# model = DenseID(classes = people_num)
denseid = 2048
classes = people_num

base_model = densenet.DenseNet121(include_top=False, pooling="avg")
x = base_model.output
x = Dense(denseid, activation="relu", name="denseid")(x)
x = Dense(classes, activation="softmax", name="fc")(x)
model = Model(inputs=base_model.input, outputs=x, name="DenseID")

for layer in base_model.layers:
    layer.trainable = False

opt = Adam(lr=0.001)
model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

np.random.seed(1024)
for _ in range(epochs):
    print("Epoch: %d/%d" % ((_+1), epochs))
    for step in range(steps_per_epoch):
        train_batch = train_data[step*batch_size:(step+1)*batch_size]
        data_batch = lfw.read_batch_file(train_batch)
        label_batch = train_label[step*batch_size:(step+1)*batch_size]
        label_batch = to_categorical(label_batch, people_num)
        hist = model.train_on_batch(data_batch, label_batch)
        val_choice = np.random.choice(len(val_label),
                                      batch_size, replace=False)
        val_batch = slice_by_list(val_data, val_choice)
        val_batch_data = lfw.read_batch_file(val_batch)
        val_batch_label = slice_by_list(val_label, val_choice)
        val_batch_label = to_categorical(val_batch_label, people_num)
        val_hist = model.test_on_batch(val_batch_data, val_batch_label)

        print("Epoch: %d, Step: %d, loss: %f, tr acc: %0.2f%%, val loss: %f, val acc: %0.2f%%" % ((_+1), (steps_per_epoch*_)+step+1, hist[0], hist[1]*100, val_hist[0], val_hist[1]*100))

    np.random.shuffle(train_data)
    train_data, train_label = my_shuffle(train_data, train_label)
    if ((_+1)%10) == 0:
        model.save("./model_save/denseid_model_"+str(_+1).zfill(4)+".h5")

model.save("./model_save/denseid_1st_stage.h5")
