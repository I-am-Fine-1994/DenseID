# 该类用于管理 http://conradsanderson.id.au/lfwcrop/ 的数据
# this class is built for management of data from
# http://conradsanderson.id.au/lfwcrop/
import tensorflow as tf
import numpy as np
import cv2 as cv # this is opencv-python module, not the official module
import os.path
import os
import sys
from functools import wraps

# folder structure looks like this:
# D:/Database/lfw/lfwcrop_color
# --faces
# --lists

def log(func):
    @wraps(func)
    def wrapper(*args, **kw):
        print("runing %s" % func.__name__)
        return func(*args, **kw)
    return wrapper

class crop_color:
    """This class provides some APIs to process LFWcrop dataseset.


    APIs:
    get_train_list_with_label()
        read dv_train_same.txt and dv_train_test.txt, and add label,
        return a list
    get_test_list_with_label()
        read dv_test_same.txt and dv_test_test.txt, and add label,
        return a list
    get_full_path_train_list_with_label()
        call get_train_list_with_label(), compelete the path of each file,
        return a list
    get_full_path_test_list_with_label()
        call get_test_list_with_label(), compelete the path of each file,
        return a list
    trainset2tfrecords()
        convert the data of images in trainset to tfrecords file
    testset2tfrecords()
        convert the data of images in testset to tfrecords file
    get_train_data_with_label(flatten=False)
        read image files in trainset as numpy array
        if flattern is True, image data array will collapse into one dimension
        return a numpy array
    get_train_data_with_label(flatten=False)
        read image files in trainset as numpy array
        if flattern is True, image data array will collapse into one dimension
        return a numpy array"""


    def __init__(self):
        # self.data_path = "/home/fine/mycode/lfw/lfwcrop_color"
        self.data_path = "D:/Database/lfwcrop_grey"
        self.list_path = "/lists"
        self.img_path = "/faces"
        self.train_list = ["/lists/dv_train_same.txt", "/lists/dv_train_diff.txt"]
        # self.train_list = ["/lists/dv_train_diff.txt", "/lists/dv_train_same.txt"]
        self.test_list = ["/lists/dv_test_same.txt", "/lists/dv_test_diff.txt"]
        # self.test_list = ["/lists/dv_test_diff.txt", "/lists/dv_test_same.txt"]
        self.lfw_names = "/lists/lfw-names.txt"
        self.file_type = ".pgm"
        # 图片的属性
        self.img_width = 64
        self.img_height = 64
        self.img_depth = 3

    # 获取训练集文件名队列
    # get the list of training set
    def get_train_list_with_label(self):
        return self.get_list_with_label(self.train_list)

    # 获取测试集文件名队列
    # get the lsit of testing set
    def get_test_list_with_label(self):
        return self.get_list_with_label(self.test_list)

    # 获取包含完整路径及标签的训练集文件名队列
    def get_full_path_train_list_with_label(self):
        return self.full_path_with_label(self.get_train_list_with_label())

    # 获取包含完整路径及标签的测试集文件名队列
    def get_full_path_test_list_with_label(self):
        return self.full_path_with_label(self.get_test_list_with_label())

    # 将训练集数据转为tfrecords文件
    def trainset2tfrecords(self):
        self.trans2tfrecords("trainset", self.get_train_list_with_label())
    # 将测试集数据转为tfrecords文件
    def testset2tfrecords(self):
        self.trans2tfrecords("testset", self.get_test_list_with_label())
    # 将训练数据读入内存
    def get_train_data_with_label(self, flatten=False):
        return self.get_image_data_with_label(
            self.get_full_path_train_list_with_label(),flatten)
    # 将测试数据读入内存
    def get_test_data_with_label(self, flatten=False):
        return self.get_image_data_with_label(
            self.get_full_path_test_list_with_label(),flatten)

    # 根据提供的文件路径列表，将数据载入到内存中
    def get_image_data_with_label(self, full_path_image_lst, flatten=False):
        img_lst = full_path_image_lst
        pair_num = len(img_lst)
        if flatten == True:
            x = np.zeros(
                [pair_num, 2, self.img_width*self.img_width*self.img_depth],
                 dtype="float32")
        else:
            x = np.zeros(
                [pair_num, 2, self.img_width, self.img_height, self.img_depth],
                 dtype="float32")
        y = np.zeros([pair_num, 1], dtype="float32")
        for i in range(pair_num):
            img1_path = img_lst[i][0]
            img2_path = img_lst[i][1]
            img1_data = cv.imread(img1_path)
            img2_data = cv.imread(img2_path)
            # if sample_rate != 1:
            #     img1_data = img1_data[::sample_rate, ::sample_rate, ::]
            #     img2_data = img2_data[::sample_rate, ::sample_rate, ::]
            if flatten == True:
                img1_data = img1_data.flatten()
                img2_data = img2_data.flatten()
            x[i, 0] = img1_data
            x[i, 1] = img2_data
            y[i] = img_lst[i][2]
        return x, y

    # 获取所有图片对的名字，并添加标签
    # 根据顺序，第一个，即same文件的标签为0，第二个diff文件的标签为1
    # 返回一个二维的列表
    # get the list of all image pairs' name, and add lable
    # according to the order, "same" will be labeled 0, "diff" will be 1
    # return a 2-D list
    def get_list_with_label(self, file_list):
        lst_lbl = []
        for i in range(len(file_list)):
            lst_lbl.extend(self.add_label(self.get_pair_list(file_list[i]),i))
        return lst_lbl

    # 给图片对增加标签
    def add_label(self, pair_list, label):
        for each in pair_list:
            each.append(label)
            # print(each)
        return pair_list

    # 读取文本文件，将每一行分割为两个元素，组成一个列表
    # 返回该列表
    # read txt file line by line, split each line into two elements as a list
    # return this list
    def get_pair_list(self, txt_path):
        pair_list = []
        load_path = self.data_path + txt_path
        print(load_path)
        with open(load_path) as f:
            lines=f.readlines()
            for line in lines:
                pair_name = line.split()
                pair_list.append(pair_name)
        return pair_list

    # 给每一条数据添加完整路径
    def full_path_with_label(self, list_with_label):
        for each in list_with_label:
            each[0] = self.data_path + self.img_path  + "/" + each[0] + self.file_type
            each[1] = self.data_path + self.img_path  + "/" + each[1] + self.file_type
        return list_with_label

    # 将数据转为tfrecords格式存储
    # trans data into tfrecords format
    def trans2tfrecords(self, recordname, filename_list):
        # 指定写入数据的文件名
        recordname = os.path.join(os.getcwd(), recordname+".tfrecords")
        print("Writing", recordname)
        # 创建写入器对象
        writer = tf.python_io.TFRecordWriter(recordname)
        # 开始写入
        for index in range(len(filename_list)):
            # print(filename_list[index])
            # print(filename_list[index][0])
            # print(os.path.join(self.data_path+self.img_path, filename_list[index][0]+"ppm"))
            img0 = cv.imread(os.path.join(self.data_path+self.img_path, \
                             filename_list[index][0]+self.file_type)).tostring()
            img1 = cv.imread(os.path.join(self.data_path+self.img_path, \
                             filename_list[index][1]+self.file_type)).tostring()
            label = filename_list[index][2]
            example = tf.train.Example(features=tf.train.Features(feature={\
                # 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.img_width])),\
                # 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.img_height])),\
                # 'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.img_depth])),\
                'img0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img0])),\
                'img1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1])),\
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            writer.write(example.SerializeToString())
        writer.close()

    # 读取tfrecords的数据
    def readtfrecords(self, recordname):
        recordname = os.path.join(os.getcwd(), recordname+".tfrecords")
        print("Reading", recordname)
        # 想要读取数据需要先将文件名转为一个队列类型
        recordname_queue = tf.train.string_input_producer([recordname])
        # 创建读取器对象
        reader = tf.TFRecordReader()
        # 读取器读取，返回(key, value)对，key个人猜测为文件名，value为文件中的内容
        _, se_exp = reader.read(recordname_queue)
        features = tf.parse_single_example(se_exp, features={\
            # 'width': tf.FixedLenFeature([], tf.int64),\
            # 'height': tf.FixedLenFeature([], tf.int64),\
            # 'depth': tf.FixedLenFeature([], tf.int64),\
            'img0': tf.FixedLenFeature([], tf.string),\
            'img1': tf.FixedLenFeature([], tf.string),\
            'label': tf.FixedLenFeature([], tf.int64)})
        print("Decoding", recordname)
        # 通过对文件内容进行解析获取其中存储的数据
        # img_width = tf.cast(features['width'], tf.int64)
        # img_height = tf.cast(features['height'], tf.int64)
        # img_depth = tf.cast(features['depth'], tf.int64)
        img0 = tf.decode_raw(features['img0'], tf.uint8)
        img0 = tf.cast(img0, tf.float32)
        img0 = tf.reshape(img0, [self.img_width, self.img_height, self.img_depth])
        img1 = tf.decode_raw(features['img1'], tf.uint8)
        img1 = tf.reshape(img0, [self.img_width, self.img_height, self.img_depth])
        img1 = tf.cast(img0, tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return img0, img1, label

if __name__ == "__main__":
    cc = crop_color()
    # lst = cc.get_full_path_test_list_with_label()
    # x, y = cc.get_train_data_with_label()
    # x_, y_ = cc.get_test_data_with_label()
    # print(x.shape)
    # print(y.shape)
    # print(x_.shape)
    # print(y_.shape)
    for each in cc.get_test_list_with_label():
        print(each)
    input("Press any key...")
    # for each in y:
        # print(each)
    # cc.get_train_list_with_label()
    # print(sys.getsizeof(cc.img_width))
    # cc.trainset2tfrecords()
    # cc.testset2tfrecords()
    # img0_data, img1_data, label_data = cc.readtfrecords("trainset")
    # img0, img1, label = tf.train.shuffle_batch([img0_data, img1_data, label_data], 2, 100, 20)
    # with tf.Session() as sess:
    #     # 创建协调器管理线程
    #     coord = tf.train.Coordinator()
    #     # 让文件名进入队列
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     i0, i1, l= sess.run([img0, img1, label])
    #     # print([i0, i1, l])
    #     coord.request_stop()
    #     coord.join(threads)