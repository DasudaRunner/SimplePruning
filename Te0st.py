#coding:utf-8
'''
Author:Wang Haibo
At: 
Email: haibo.david@qq.com
'''

from pruner import Pruner
import tensorflow as tf

model = Pruner()

a = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)

