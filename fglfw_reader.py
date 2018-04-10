import numpy as np
import cv2 as cv
import os.path
import os
import sys

class fglfw_reader():

    def __init__(self):
        self.data_path = "D:/Database/lfwcrop_color"
        self.list_path = "/lists"
        self.img_path = "/faces"
        self.txt_name = "/pair_FGLFW.txt"

        # 图片属性
        self.img_width = 64
        self.img_height = 64
        self.img_depth = 3

    # 将txt内容存放到列表中
    def parse_txt_to_list(self):
        file_path = self.data_path + self.list_path + self.txt_name
        label = 0
        pair_list = []
        pair = []
        with open(file_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # 每三百行标签修改一次
                if i%300 == 0:
                    label = (i//300)%2
                # 每两行组成一个pair
                pair.append(line.strip())
                if (i+1)%2 == 0:
                    # 添加标签
                    pair.append(label)
                    pair_list .append(pair)
                    pair = []
        return pair_list

    def get_pairs(self, ext=".ppm"):
        file_list = self.parse_txt_to_list()
        file_list = self.fix_list_type(file_list, ext)
        file_list = self.reserve_list_filename(file_list)
        return file_list

    def get_full_path_pairs(self):
        file_list = self.get_pairs()
        for each in file_list:
            each[0] = self.data_path + self.img_path  + "/" + each[0]
            each[1] = self.data_path + self.img_path  + "/" + each[1]
        return file_list

    def get_data(self, flatten=False):
        print("loading FGLFW data...")
        img_lst = self.get_full_path_pairs()
        pair_num = len(img_lst)
        if flatten == True:
            x = np.zeros([pair_num, 2, self.img_width*self.img_width*self.img_depth], dtype="float32")
        else:
            x = np.zeros([pair_num, 2, self.img_width, self.img_height, self.img_depth], dtype="float32")
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

    # 修改FGLFW提供的txt中所有的文件名的后缀
    def fix_list_type(self, file_list, ext):
        # file_list = self.parse_txt_to_list()
        for each in file_list:
            each[0] = self.fix_type(each[0], ext)
            each[1] = self.fix_type(each[1], ext)
        return file_list

    # 去掉FGLFW提供的txt中，所有的文件名的路径，只保留文件名
    def reserve_list_filename(self, file_list):
        # file_list = self.parse_txt_to_list()
        for each in file_list:
            each[0] = self.reserve_filename(each[0])
            each[1] = self.reserve_filename(each[1])
        return file_list

    # 修改文件名的后缀
    def fix_type(self, file_name, ext):
        # 将后缀名与文件名分开
        file_name = os.path.splitext(file_name)[0]
        # 为文件名添加新的后缀名
        file_name = file_name + ext
        return file_name

    # 去掉路径，只保留文件名
    def reserve_filename(self, file_path):
        # 将/前的部分与文件名分开
        file_name = os.path.split(file_path)[-1]
        return file_name

if __name__ == "__main__":
    fr = fglfw_reader()
    # pairs = fr.parse_txt_to_list()
    # for each in pairs:
    #     print(each)
    x, y = fr.get_data()
    print(x.shape)
    print(y.shape)
    input("press enter...")
    # print((int)True)
    # print((int)False)