#coding:utf-8
#author:Wang Haibo
#at: Pingan Tec.

import cv2
import pickle as pkl
import numpy as np
import os
import tensorflow as tf
import random

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']

def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)

class CifarData:
    def __init__(self,filenames, need_shuffle):
        all_data = []
        all_labels = []

        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        self._data = np.vstack(all_data)
        self._data = self._data/127.5-1.
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    # 获得一个boundary/total的概率事件是否发生
    def __roll_dice(self,total,boundary):
        if boundary>=total or total<=1 :
            raise ValueError("in __roll_dice, boundary cannot bigger than total. boundary must be bigger than 1.")

        dice_val = random.randint(1,total)
        if dice_val<=boundary:
            return True
        else:
            return False

    def __data_augmentation(self,batch_imgs):

        new_batch_imgs = []
        for i in range(batch_imgs.shape[0]):
            out = batch_imgs[i].copy()
            if self.__roll_dice(3, 2):
                img_h = out.shape[0]
                img_w = out.shape[1]
                if True or self.__roll_dice(3,2): #这个一直执  行
                    out = cv2.flip(out,0) #left right
                    # out = cv2.flip(batch_imgs[i],0) #up down
                    # out = cv2.flip(batch_imgs[i],-1) #both
                if self.__roll_dice(2,1): # 裁剪
                    zoom_w = 4
                    zoom_h = 4
                    out = cv2.resize(out,(img_h+zoom_h,img_w+zoom_w))
                    out = out[zoom_h//2:zoom_h//2+img_h,zoom_w//2:zoom_w//2+img_w,:]
                if False and self.__roll_dice(2,1): #噪声
                    noise = np.random.normal(0, 0.01 ** 0.5, out.shape)
                    out += noise
                    out = np.clip(out,-1.0,1.0)
                if self.__roll_dice(2,1): #旋转
                    random_angle = random.randint(30,90)
                    rotate_center = (img_w//2,img_h//2)
                    M = cv2.getRotationMatrix2D(rotate_center, random_angle, 1.0)
                    out = cv2.warpAffine(out, M, (img_w, img_h))

            new_batch_imgs.append(out)

        else:
            return batch_imgs

    def next_batch(self, batch_size):
        end_indictor = self._indicator + batch_size
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indictor = batch_size
            else:
                raise Exception("have no more examples")

        if end_indictor > self._num_examples:
            raise Exception("batch size is larger than all example")

        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor

        batch_data = np.reshape(batch_data, (-1, 3, 32, 32))
        batch_data = np.transpose(batch_data, (0, 2, 3, 1))
        batch_data = self.__data_augmentation(batch_data)

        batch_labels = np.array(batch_labels).astype(np.int32)
        # # batch_labels = make_one_hot(batch_labels)

        return batch_data, batch_labels

if __name__ == "__main__":
    # # 文件存放目录
    # CIFAR_DIR = "./cifar-10-python"
    #
    # train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    # test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
    # train_data = CifarData(train_filename, True)
    # test_data = CifarData(test_filename, True)
    #
    # x,y = train_data.next_batch(16)
    #
    # for i in range(16):
    #     img = (x[i]+1.)*127.5
    #     img = img.astype(np.uint8)
    #
    #     cv2.imshow("src",img)
    #
    #     cv2.waitKey(0)
    # print(y)

    # 文件存放目录
    CIFAR_DIR = "./cifar-100-python"

    train_filename = ["./cifar-100-python/train"]
    test_filename = ["./cifar-100-python/test"]
    train_data = CifarData(train_filename, True,"cifar100")
    test_data = CifarData(test_filename, True,"cifar100")

    x,y = train_data.next_batch(16)

    for i in range(16):
        img = (x[i]+1.)*127.5
        img = img.astype(np.uint8)

        cv2.imshow("src",img)

        cv2.waitKey(0)
    print(y)