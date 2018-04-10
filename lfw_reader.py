import numpy as np
import os.path
import os
import sys
import cv2 as cv
import platform

class lfw_reader():

    def __init__(self):
        if platform.system() == "Windows":
            self.crop_data_path = "D:\Database\lfwcrop_color"
            self.data_path = "D:\Database\lfw"
        if platform.system() == "Linux":
            self.data_path = "/home/x000000/LK/Database/lfw"
            self.crop_data_path = "/home/x000000/LK/Database/lfwcrop_color"

        self.img_path = "faces"
        self.list_path = "lists"

        self.pairs_dev_train = "pairsDevTrain.txt"
        self.pairs_dev_test = "pairsDevTest.txt"
        self.people_dev_train = "peopleDevTrain.txt"
        self.people_dev_test = "peopleDevTest.txt"
        self.pairs = "pairs.txt"
        self.people = "people.txt"
        self.lfw_names = "lfw-names.txt"

        self.img_width = 250
        self.img_height = 250
        self.img_depth = 3

        self.names = self.list_names()
        self.name_idx = self.dict_name_idx()

    # 获取所有名字
    def list_names(self):
        file_path = os.path.join(self.data_path,
                                 self.list_path,
                                 self.lfw_names)
        names = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                names.append(line.split()[0])
        return names

    # 根据名字的读取顺序确定标签，读取顺序按照lfw-names.txt，
    # 用字典将名字和标签对应，返回该字典
    def dict_name_idx(self):
        dict_name_id = {}
        for i, name in enumerate(self.names):
            dict_name_id[name] = i
        return dict_name_id

    # 返回一个生成器，返回全部图片的完整路径和标签
    def all_img_full_path(self):
        file_path = os.path.join(self.data_path, self.img_path)
        for name in self.names:
            ind_path = os.path.join(file_path, name)
            for per_img in os.listdir(ind_path):
                per_img_full_path = os.path.join(ind_path, per_img)
                yield per_img_full_path, self.name_idx[name]

    # 到lfwcrop文件夹中去读取人脸的数据
    def load_lfwcrop_data(self, flatten=False):
        print("loading lfwcrop...")
        lfwc_data_path = self.crop_data_path
        lfwc_img_path = os.path.join(lfwc_data_path, self.img_path)
        data = np.zeros([13233, 64, 64, 3], dtype="float32")
        label = np.zeros([13233, 1], dtype="float32")
        i = 0
        for per_img in os.listdir(lfwc_img_path):
            path = os.path.join(lfwc_img_path, per_img)
            idx = self.name_idx[per_img[:-9]]
            # print(path, idx)
            img_data = cv.imread(path)
            data[i] = img_data
            label[i] = idx
            i += 1
        print("loading compeleted.")
        return data, label

    # 根据给定的一组下标读取图片，lfwcrop_color/faces文件夹下有13233张图片，
    # 以第一张图片下标为0，最后一张图片下标为13232。
    # 返回值为图片数据和对应的标签
    def load_lfwcrop_data_batch(self, indexes:list):
        lfwc_data_path = self.crop_data_path
        lfwc_img_path = os.path.join(lfwc_data_path, self.img_path)
        dir_file = os.listdir(lfwc_img_path)
        # indexes = np.arange(len(dir_file))
        idx_file = []
        for idx in indexes:
            idx_file.append(dir_file[idx])
        data = np.zeros([len(indexes), 64, 64, 3], dtype="float32")
        label = np.zeros([len(indexes)], dtype="int32")
        i = 0
        for per_img in idx_file:
            path = os.path.join(lfwc_img_path, per_img)
            idx = self.name_idx[per_img[:-9]]
            # print(path, idx)
            img_data = cv.imread(path)
            data[i] = img_data
            label[i] = idx
            i += 1
        return data, label

    # 查找图片数量不少于num的人并将名字加入列表中，返回列表
    def no_less_than(self, num: int) -> list:
        name_lst = []
        file_path = os.path.join(self.data_path, self.img_path)
        for indv_folder in os.listdir(file_path):
            file_lst = os.listdir(os.path.join(file_path, indv_folder))
            if len(file_lst) >= num:
                name_lst.append(indv_folder)
        return name_lst

    # 从图片数量大于 num 的人中，抽取 num 张图片，生成图片在 lfwcrop 数据集中的
    # 完整路径和标签，返回路径 list 和 标签 list
    def form_data_from_nlt(self, num:int):
        name_lst = self.no_less_than(num)
        lfwc_data_path = self.crop_data_path
        lfwc_img_path = os.path.join(lfwc_data_path, self.img_path)
        path_data = []
        path_label = []
        for label, name in enumerate(name_lst):
            # print(label, name)
            file_path = os.path.join(self.data_path, self.img_path, name)
            nb = len(os.listdir(file_path))
            # print(nb)
            np.random.seed(1024)
            # 每个人抽取 num 张图片，replace 为否，不可以有重复的下标
            idx = np.random.choice(nb, num, replace=False)
            idx += 1
            # print(idx)
            for each in idx:
                img_file = name + "_" + str(each).zfill(4) + ".ppm"
                img_file = os.path.join(lfwc_img_path, img_file)
                # print(img_file, label)
                path_data.append(img_file)
                path_label.append(label)
        return path_data, path_label

    def read_batch_file(self, file_name_lst):
        batch_data = np.zeros([len(file_name_lst), 64, 64, 3], dtype="float32")
        for i, per_img in enumerate(file_name_lst):
            img_data = cv.imread(per_img)
            # if img_data is not None:
                # print("reading %s" % per_img)
            batch_data[i] = img_data
        return batch_data

if __name__ == "__main__":
    lfw = lfw_reader()
    # lfw.form_data_from_nlt(10)
    # lfw.load_lfwcrop_data()
    # try split dataset 2 test, 2 val, 4 train
    # lst = lfw.no_less_than(10)