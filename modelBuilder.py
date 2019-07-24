#coding:utf-8

'''
author:Wang Haibo
at: Pingan Tec.
email: haibo.david@qq.com

!!!
代码中会有少量中文注释，无需在意

'''

import os
import time
import numpy as np
import tensorflow as tf
from datasets import CifarData

class Cifar10ModelBuilder:
    def __init__(self,ckpt_path,input_node,is_train_node,output_node):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(ckpt_path+".meta")
            self.sess = tf.Session(graph=self.graph)
            with self.sess.as_default():
                with self.graph.as_default():
                    self.saver.restore(self.sess, ckpt_path)
        self.input_x = self.sess.graph.get_tensor_by_name(input_node+":0")
        self.is_train = self.sess.graph.get_tensor_by_name(is_train_node+":0")
        self.output= self.sess.graph.get_tensor_by_name(output_node+":0")

    def predict(self,data):
        out = self.sess.run(self.output,feed_dict={self.input_x:data,self.is_train:False})
        return out

    def close(self):
        self.sess.close()

def test_full_model():
    # 文件存放目录
    CIFAR_DIR = "./cifar-10-python"
    test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
    test_data = CifarData(test_filename, True)

    test_batch_size = 64
    test_batches = test_data._num_examples // test_batch_size

    model_builder = Cifar10ModelBuilder("./ckpt_model/full_model.ckpt",
                                        "input0","is_training","softmax")
    all_acc = []
    all_t = []
    for i in range(test_batches):
        batch_data, batch_labels = test_data.next_batch(test_batch_size)
        t_start = time.time()
        res = model_builder.predict(batch_data)
        all_t.append(float(time.time() - t_start) * 1000.)
        correct = np.equal(np.argmax(res,1),batch_labels).astype(np.float32)
        acc = np.mean(correct)
        all_acc.append(acc)
    model_builder.close()
    print("[full model]: acc: %4.5f, time/ms: %4.5f" % (np.mean(all_acc),np.mean(all_t)))
