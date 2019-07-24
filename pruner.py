#coding:utf-8

'''
author:Wang Haibo
at: Pingan Tec.
email: haibo.david@qq.com

!!!
代码中会有少量中文注释，无需在意

'''

import numpy as np
import pickle as pkl
import tensorflow as tf
import os
from tensorflow.contrib.layers import variance_scaling_initializer as msra_init

class Pruner():
    def __init__(self,reload_file=None):

        self.__model_scope = "full_model"

        self.__weights={}
        self.__pruning_mask={} # weights prune
        self.__chanels_mask={} # channels prune
        self.__add_shape={} # special for add op

        self.__layer_cnt=0

        self.__support_ops=["conv","fc","dconv"]
        self.__support_special_ops = ["add"]

        self.__support_prune_weights_type = ["rate","threshold"]
        self.__support_prune_channels_type = ["rate","auto"]

        self.__layer_remap = {} # 记录网络的拓扑结构
        self.__index_remap = [] # __layer_remap中为self.__support_ops的layer

        self.__shapes_data = None
        if reload_file is not None:
            with open(reload_file, "rb") as f:
                self.__shapes_data = pkl.load(f)

    def __next_iteration(self):
        self.__layer_cnt+=1

    # total number of layers
    @property
    def __total_w_count(self):
        return len(self.__index_remap)

    def print_layer_remap(self):
        print(self.__layer_remap)
        print(self.__index_remap)
        print("trainable weights:",len(self.__weights.keys()))
        print(self.__weights.keys())

    # add a layer using my api,you can find support layers in self.__support_ops.
    def _add_layer(self,
                   inputs=None,
                   mode='conv',
                   out_c=16,
                   k_size=3,
                   strides=1,
                   padding='SAME',
                   act=tf.nn.relu,
                   with_bn=False,
                   is_train=None,
                   name=None):
        '''
        Args:
            input: input tensor
            model: suported ops in self.__support_ops
            out_c: output channels
            k_size: only valid in convolution.
            strides: strides
            padding: padding
            act: activation funciton
            with_bn: use bn layer or not
            is_train: when you set with_bn is True, you must set this para.
            name: None, not valid
            ...

        Return:
            output tensor

        Raise:
            ValueError("_add_layer() "+mode+" is not supported!")
            ValueError("bn layer must be with relu.")
            ValueError("when using bn layer, you must be set is_train attr.")
            ValueError("when using dconv layer, you must be set is_train attr.")

        '''

        if mode not in self.__support_ops:
            raise ValueError("_add_layer() "+mode+" is not supported!")

        if with_bn is True and act is False:
            raise ValueError("bn layer must be with relu.")

        if with_bn is True and is_train is None:
            raise ValueError("when using bn layer, you must be set is_train attr.")

        if mode=="dconv" and is_train is None:
            raise ValueError("when using dconv layer, you must be set is_train attr.")

        in_c = inputs.get_shape().as_list()[-1]

        if mode=="fc":

            if self.__shapes_data is None:
                temp_b = tf.Variable(tf.constant(value=0.01, shape=[out_c]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w = tf.get_variable(name="w_" + str(self.__layer_cnt),shape=[in_c, out_c],dtype=tf.float32,
                                         initializer=msra_init())
            else:
                temp_shape = self.__shapes_data["fc_" + str(self.__layer_cnt)]
                temp_b = tf.Variable(tf.constant(value=0.01, shape=[temp_shape]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w = tf.get_variable(name="w_" + str(self.__layer_cnt), shape=[in_c, temp_shape], dtype=tf.float32,
                                         initializer=msra_init())

            temp_op = tf.nn.bias_add(tf.matmul(inputs,temp_w),temp_b,name="fc_" + str(self.__layer_cnt))

            self.__insert_remap(temp_op.name[:-2], [inputs.name[:-2]])

        elif mode=="conv":

            if self.__shapes_data is None:
                temp_b = tf.Variable(tf.constant(value=0.01, shape=[out_c]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w = tf.get_variable(name="w_" + str(self.__layer_cnt), shape=[k_size, k_size, in_c, out_c], dtype=tf.float32,
                                         initializer=msra_init())
            else:
                temp_shape = self.__shapes_data["conv_" + str(self.__layer_cnt)]
                temp_b = tf.Variable(tf.constant(value=0.01, shape=[temp_shape]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w = tf.get_variable(name="w_" + str(self.__layer_cnt), shape=[k_size, k_size, in_c, temp_shape], dtype=tf.float32,
                                         initializer=msra_init())

            temp_op = tf.nn.bias_add(tf.nn.conv2d(inputs, filter=temp_w, strides=[1,strides,strides,1], padding=padding),
                                     temp_b, name="conv_" + str(self.__layer_cnt))

            self.__insert_remap(temp_op.name[:-2], [inputs.name[:-2]])

        elif mode=="dconv":

            if self.__shapes_data is None:

                temp_b = tf.Variable(tf.constant(value=0.01, shape=[out_c]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w_dep = tf.get_variable(name="w_"+str(self.__layer_cnt)+"_dep",shape=[k_size, k_size, in_c, 1],dtype=tf.float32,
                                             initializer=msra_init())
                temp_w = tf.get_variable(name="w_"+str(self.__layer_cnt)+"_pts",shape=[1, 1, in_c, out_c],dtype=tf.float32,
                                             initializer=msra_init())
            else:

                temp_shape = self.__shapes_data["dconv_"+str(self.__layer_cnt)]
                temp_b = tf.Variable(tf.constant(value=0.01, shape=[temp_shape]),
                                     name="b_" + str(self.__layer_cnt))
                temp_w_dep = tf.get_variable(name="w_"+str(self.__layer_cnt)+"_dep",shape=[k_size, k_size, in_c, 1],dtype=tf.float32,
                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
                temp_w = tf.get_variable(name="w_"+str(self.__layer_cnt)+"_pts",shape=[1, 1, in_c, temp_shape],dtype=tf.float32,
                                             initializer=msra_init())

            # temp_op = tf.nn.bias_add(tf.nn.separable_conv2d(inputs,temp_w_dep,temp_w,strides=[1,strides,strides,1],padding=padding),
            #                          temp_b, name="dconv_"+str(self.__layer_cnt))
            temp_op_1 = tf.nn.depthwise_conv2d(inputs,temp_w_dep,strides=[1,strides,strides,1],padding=padding)
            temp_op_1 = tf.layers.batch_normalization(temp_op_1,training=is_train)
            temp_op_1 = act(temp_op_1)
            temp_op = tf.nn.bias_add(tf.nn.conv2d(temp_op_1,filter=temp_w,strides=[1,1,1,1],padding="SAME"),
                                     temp_b,name="dconv_"+str(self.__layer_cnt))

            self.__insert_remap(temp_op.name[:-2], [inputs.name[:-2]])

        if with_bn:
            out1 = tf.layers.batch_normalization(temp_op,training=is_train)
            out = act(out1, name="relu_" + str(self.__layer_cnt))
            self.__insert_remap(out.name[:-2], [temp_op.name[:-2]])
        else:
            if act is not None:
                out = act(temp_op,name="relu_"+str(self.__layer_cnt))
                self.__insert_remap(out.name[:-2], [temp_op.name[:-2]])
            else:
                out = temp_op

        if mode=="dconv":
            self.__weights.update({"w_" + str(self.__layer_cnt)+"_dep": temp_w_dep})
            self.__weights.update({"w_" + str(self.__layer_cnt)+"_pts": temp_w})
            self.__weights.update({"b_" + str(self.__layer_cnt): temp_b})
        else:
            self.__weights.update({"w_" + str(self.__layer_cnt): temp_w})
            self.__weights.update({"b_" + str(self.__layer_cnt): temp_b})

        self.__next_iteration()

        return out

    def bn_act_layer(self,inputs,is_train,act=tf.nn.relu):
        out = tf.layers.batch_normalization(inputs, training=is_train)
        out = act(out, name="relu_" + str(self.__layer_cnt))
        self.__insert_remap(out.name[:-2], [inputs.name[:-2]])
        self.__next_iteration()
        return out

    def pool_layer(self,inputs,mode="max",pool_size=2,strides=2,padding="SAME"):
        if mode=="max":
            pool_func = tf.nn.max_pool
            op_name = "maxpool_"+str(self.__layer_cnt)
        elif mode=="avg":
            pool_func = tf.nn.avg_pool
            op_name = "avgpool_" + str(self.__layer_cnt)
        else:
            raise ValueError("mode must be max or avg in pool_layer()")

        temp_op = pool_func(inputs,ksize=[1,pool_size,pool_size,1],strides=[1,strides,strides,1],
                                 padding=padding,name=op_name)
        self.__insert_remap(op_name, [inputs.name[:-2]])
        self.__next_iteration()
        return temp_op

    def flatten_layer(self,inputs):
        temp_op = tf.reshape(inputs,[-1,int(inputs.get_shape().as_list()[-1])],name="flat_"+str(self.__layer_cnt))
        self.__insert_remap("flat_" + str(self.__layer_cnt), [inputs.name[:-2]])
        self.__next_iteration()
        return temp_op

    def gmp_layer(self,inputs):
        pool_size = inputs.get_shape().as_list()[1]
        temp_op = tf.nn.max_pool(inputs,ksize=[1,pool_size,pool_size,1],strides=[1,1,1,1],
                                 padding="VALID")
        out = tf.reshape(temp_op, [-1, int(temp_op.get_shape().as_list()[-1])], name="gam_" + str(self.__layer_cnt))
        self.__insert_remap("gam_" + str(self.__layer_cnt), [inputs.name[:-2]])
        self.__next_iteration()
        return out

    def gap_layer(self,inputs):
        pool_size = inputs.get_shape().as_list()[1]
        temp_op = tf.nn.avg_pool(inputs,ksize=[1,pool_size,pool_size,1],strides=[1,1,1,1],
                                 padding="VALID")
        out = tf.reshape(temp_op, [-1, int(temp_op.get_shape().as_list()[-1])],name="gap_"+str(self.__layer_cnt))
        self.__insert_remap("gap_" + str(self.__layer_cnt), [inputs.name[:-2]])
        self.__next_iteration()
        return out

    def concat_layer(self,inputs,concat_dims=3):
        if not isinstance(inputs,list):
            raise TypeError("inputs must be list in concat_layer()")

        temp_op = tf.concat(inputs,concat_dims,name="concat_"+str(self.__layer_cnt))

        self.__insert_remap("concat_" + str(self.__layer_cnt), [x.name[:-2] for x in inputs])
        self.__next_iteration()
        return temp_op

    def Add_layer(self,a,b):

        temp_op = tf.add(a,b,name="add_"+str(self.__layer_cnt))

        self.__add_shape.update({temp_op.name[:-2]:a.get_shape().as_list()[-1]})
        self.__insert_remap(temp_op.name[:-2], [a.name[:-2],b.name[:-2]])
        self.__next_iteration()
        return temp_op

    def __insert_remap(self,keys,val):
        if len(self.__layer_remap.values())==0:
            self.__layer_remap.update({keys:None})
        else:
            self.__layer_remap.update({keys:val})

        op_details = keys.split("_")
        if op_details[0] in self.__support_ops:
            self.__index_remap.append(keys)

    # do not used!!!
    def KL_Divergence(self,p_dis,q_dis):
        if len(p_dis)!=len(q_dis):
            raise ValueError("KL_Divergence calc error.")
        total_len = len(p_dis)
        KL = 0
        for i in range(total_len):
            if q_dis[i]==0.0:
                KL += p_dis[i]*np.log(p_dis[i]/(q_dis[i]+0.0001))
            else:
                KL += p_dis[i]*np.log(p_dis[i]/q_dis[i])

        return KL

    # return a list, even though one father valid.
    def __get_father(self,son_name):
        real_father=[]
        while True:
            temp_f = self.__layer_remap[son_name]

            if len(temp_f)>1:
                remain_ops = temp_f.copy()
                goon = True
                while goon:
                    goon = False
                    temp_ops = []
                    for i in remain_ops:
                        op_details = i.split("_")
                        if op_details[0] in (self.__support_ops+self.__support_special_ops):
                            temp_ops.append(i)
                        else:
                            goon = True
                            temp_ops+=self.__layer_remap[i]
                    remain_ops = temp_ops.copy()

                real_father = remain_ops.copy()
                break
            else:
                op_details = temp_f[0].split("_")
                if op_details[0] in (self.__support_ops+self.__support_special_ops):
                    real_father+=temp_f
                    break
                else:
                    son_name = temp_f[0]

        return real_father

    def __get_son(self,father_name):
        son_name = []

        _left_son_name = [father_name]
        whole_support_ops = self.__support_ops+self.__support_special_ops
        go_on = True
        while go_on:
            go_on = False
            _temp_left_son = []
            for s_son in _left_son_name:
                # find son nodes
                for i in self.__layer_remap.keys():
                    if self.__layer_remap[i] is None:
                        continue
                    if s_son in self.__layer_remap[i]:
                        if self.__get_op_type(i) not in whole_support_ops:
                            go_on = True
                            _temp_left_son.append(i)
                        else:
                            if i not in son_name:
                                son_name.append(i)

            _left_son_name = _temp_left_son.copy()

        return son_name

    def __get_op_type(self,full_name):
        temp = full_name.split("_")
        if len(temp)!=2:
            raise ValueError("op name error -> ",full_name)
        return temp[0]

    def __get_op_index(self,full_name):
        temp = full_name.split("_")
        if len(temp)!=2:
            raise ValueError("op name error -> ",full_name)
        return temp[1]

    def __prune_s_layer_channels(self,sess,mode,prune_rate,sl):

        op_type = self.__get_op_type(self.__index_remap[sl])
        op_index = self.__get_op_index(self.__index_remap[sl])

        if op_type=="conv": #conv

            ts_weight = self.__weights["w_" + op_index]
            ts_bias = self.__weights["b_" + op_index]
            np_weight, np_bias = sess.run([ts_weight, ts_bias])

            if len(self.__chanels_mask)>0: #pre prune
                father_name = self.__get_father(self.__index_remap[sl])
                if len(father_name)==1:
                    if not self.__get_op_type(father_name[0])=="add": #当father是add时，不进行pre pruning
                        temp_mask = self.__chanels_mask[father_name[0]]
                        np_weight = np_weight[:, :, temp_mask, :]
                else: # must be concat
                    cnt = 0
                    real_mask = []
                    for i in father_name:
                        temp_index = self.__get_op_index(i)
                        if self.__get_op_type(i)=="add":
                            total_len = self.__add_shape["add_"+temp_index]

                            single_mask = [i for i in range(total_len)] # 全部保留
                        else:
                            if self.__get_op_type(i)=="dconv":
                                total_len = self.__weights["w_" + temp_index+"_pts"].get_shape().as_list()[-1]
                            else:
                                total_len = self.__weights["w_" + temp_index].get_shape().as_list()[-1]

                            single_mask = self.__chanels_mask[i]

                        _single_mask = np.array(single_mask) + cnt
                        real_mask += list(_single_mask)

                        cnt += total_len

                    real_mask = np.array(real_mask)
                    np_weight = np_weight[:, :, real_mask, :]

            son_list = self.__get_son(self.__index_remap[sl])

            # if currnet node is the last node
            if len(son_list)==0:
                return_weights_dict = {"w_" + op_index: np_weight,
                                       "b_" + op_index: np_bias}
                return return_weights_dict, np_weight.shape

            for s_n in son_list:
                if self.__get_op_type(s_n)=="add":
                    return_weights_dict = {"w_" + op_index: np_weight,
                                           "b_" + op_index: np_bias}
                    prune_channel_index = [i for i in range(np_weight.shape[-1])]
                    self.__chanels_mask.update(
                        {self.__index_remap[sl]: prune_channel_index})  # save the channels mask
                    return return_weights_dict,np_weight.shape

            conv_sum = np.sum(np_weight,(0,1,2))
            abs_conv_sum = np.abs(conv_sum)
            sorted_conv_sum = np.sort(abs_conv_sum)

            real_prune_c = int(conv_sum.shape[0] * prune_rate)

            if real_prune_c < 1:
                real_prune_c = 1

            # get index
            prune_channel_index = np.where(abs_conv_sum >= sorted_conv_sum[real_prune_c - 1])

            # get remain channels
            pruned_channels_weights = np_weight[:, :, :, prune_channel_index[0]]
            pruned_channels_bias = np_bias[prune_channel_index[0]]

            return_weights_dict = {"w_" + op_index:pruned_channels_weights,"b_" + op_index:pruned_channels_bias}

        elif op_type=="fc":

            ts_weight = self.__weights["w_" + op_index]
            ts_bias = self.__weights["b_" + op_index]
            np_weight, np_bias = sess.run([ts_weight, ts_bias])

            if len(self.__chanels_mask)>0: #pre prune
                father_name = self.__get_father(self.__index_remap[sl])
                if len(father_name)==1:
                    if not self.__get_op_type(father_name[0]) == "add":  # 当father是add时，不进行pre pruning
                        temp_mask = self.__chanels_mask[father_name[0]]
                        np_weight = np_weight[temp_mask,:]
                else:
                    cnt = 0
                    real_mask = []
                    for i in father_name:
                        temp_index = self.__get_op_index(i)

                        if self.__get_op_type(i)=="add":
                            total_len = self.__add_shape["add_"+temp_index]

                            single_mask = [i for i in range(total_len)] # 全部保留
                        else:
                            if self.__get_op_type(i)=="dconv":
                                total_len = self.__weights["w_" + temp_index+"_pts"].get_shape().as_list()[-1]
                            else:
                                total_len = self.__weights["w_" + temp_index].get_shape().as_list()[-1]

                            single_mask = self.__chanels_mask[i]

                        _single_mask = np.array(single_mask) + cnt
                        real_mask += list(_single_mask)

                        cnt += total_len

                    real_mask = np.array(real_mask)
                    np_weight = np_weight[real_mask, :]

            son_list = self.__get_son(self.__index_remap[sl])

            # if currnet node is the last node
            if len(son_list)==0:
                return_weights_dict = {"w_" + op_index: np_weight,
                                       "b_" + op_index: np_bias}
                return return_weights_dict, np_weight.shape

            for s_n in son_list:
                if self.__get_op_type(s_n)=="add":
                    return_weights_dict = {"w_" + op_index: np_weight,
                                           "b_" + op_index: np_bias}
                    prune_channel_index = [i for i in range(np_weight.shape[-1])]
                    self.__chanels_mask.update(
                        {self.__index_remap[sl]: prune_channel_index})  # save the channels mask
                    return return_weights_dict,np_weight.shape

            conv_sum = np.sum(np_weight, (0,))
            abs_conv_sum = np.abs(conv_sum)
            sorted_conv_sum = np.sort(conv_sum)

            real_prune_c = int(conv_sum.shape[0] * prune_rate)

            if real_prune_c < 1:
                real_prune_c = 1

            # get index
            prune_channel_index = np.where(conv_sum >= sorted_conv_sum[real_prune_c - 1])

            # get remain channels
            pruned_channels_weights = np_weight[:,prune_channel_index[0]]
            pruned_channels_bias = np_bias[prune_channel_index[0]]

            return_weights_dict = {"w_" + op_index: pruned_channels_weights, "b_" + op_index: pruned_channels_bias}

        elif op_type=="dconv":

            ts_weight_dep = self.__weights["w_" + op_index+"_dep"]
            ts_weight_pts = self.__weights["w_" + op_index+"_pts"]
            ts_bias = self.__weights["b_" + op_index]
            np_weight_dep,np_weight_pts,np_bias = sess.run([ts_weight_dep, ts_weight_pts,ts_bias])

            if len(self.__chanels_mask)>0: #pre prune
                father_name = self.__get_father(self.__index_remap[sl])
                if len(father_name)==1:
                    if not self.__get_op_type(father_name[0]) == "add":  # 当father是add时，不进行pre pruning
                        temp_mask = self.__chanels_mask[father_name[0]]
                        np_weight_dep = np_weight_dep[:, :, temp_mask, :]
                        np_weight_pts = np_weight_pts[:, :, temp_mask, :]
                else:
                    cnt = 0
                    real_mask = []
                    for i in father_name:
                        temp_index = self.__get_op_index(i)

                        if self.__get_op_type(i)=="add":
                            total_len = self.__add_shape["add_"+temp_index]

                            single_mask = [i for i in range(total_len)] # 全部保留
                        else:
                            if self.__get_op_type(i)=="dconv":
                                total_len = self.__weights["w_" + temp_index+"_pts"].get_shape().as_list()[-1]
                            else:
                                total_len = self.__weights["w_" + temp_index].get_shape().as_list()[-1]

                            single_mask = self.__chanels_mask[i]

                        _single_mask = np.array(single_mask) + cnt
                        real_mask += list(_single_mask)

                        cnt += total_len

                    real_mask = np.array(real_mask)
                    np_weight_dep = np_weight_dep[:, :, real_mask, :]
                    np_weight_pts = np_weight_pts[:, :, real_mask, :]

            son_list = self.__get_son(self.__index_remap[sl])

            # if currnet node is the last node
            if len(son_list)==0:
                return_weights_dict = {"w_" + op_index + "_dep": np_weight_dep,
                                       "w_" + op_index + "_pts": np_weight_pts,
                                       "b_" + op_index: np_bias}
                return return_weights_dict, np_weight_pts.shape

            for s_n in son_list:
                if self.__get_op_type(s_n)=="add":
                    return_weights_dict = {"w_" + op_index + "_dep": np_weight_dep,
                                           "w_" + op_index + "_pts": np_weight_pts,
                                           "b_" + op_index: np_bias}
                    prune_channel_index = [i for i in range(np_weight_pts.shape[-1])]
                    self.__chanels_mask.update(
                        {self.__index_remap[sl]: prune_channel_index})  # save the channels mask
                    return return_weights_dict,np_weight.shape

            conv_sum = np.sum(np_weight_pts,(0,1,2))
            abs_conv_sum = np.abs(conv_sum)
            sorted_conv_sum = np.sort(abs_conv_sum)

            real_prune_c = int(conv_sum.shape[0] * prune_rate)

            if real_prune_c < 1:
                real_prune_c = 1

            # get index
            prune_channel_index = np.where(abs_conv_sum >= sorted_conv_sum[real_prune_c - 1])

            # get remain channels
            pruned_channels_weights = np_weight_pts[:, :, :, prune_channel_index[0]]
            pruned_channels_bias = np_bias[prune_channel_index[0]]

            return_weights_dict = {"w_" + op_index+"_dep": np_weight_dep,
                                   "w_" + op_index+"_pts": pruned_channels_weights,
                                   "b_" + op_index: pruned_channels_bias}

        self.__chanels_mask.update({self.__index_remap[sl]:prune_channel_index[0]}) # save the channels mask

        return return_weights_dict,pruned_channels_weights.shape

    def _prune_channels(self,sess,prune_mode="rate",prune_rate=0.5,verbose=True):

        if prune_mode not in self.__support_prune_channels_type:
            raise ValueError("prune_type must be ", self.__support_prune_channels_type)

        if prune_mode=="rate":
            if prune_rate<0.005 or prune_rate>1.0:
                raise ValueError("threshold is beyond [0.005, 1.0]")

        self.print_layer_remap()

        total_layers = self.__total_w_count
        saved_weights = {}
        saved_shape={}

        #finally layer,pre-prune.
        for sl in range(total_layers):

            if self.__get_op_type(self.__index_remap[sl]) in self.__support_ops:

                new_weights,new_channels = self.__prune_s_layer_channels(sess,prune_mode,prune_rate,sl)

                saved_weights.update(new_weights)

                saved_shape.update({self.__index_remap[sl]:new_channels[-1]})

                if verbose:
                    print("layer name:",self.__index_remap[sl],
                          ", pruned channels:",new_channels)

        with open("./weights_data/weights_"+str(prune_rate)+".pkl","wb") as f:
            pkl.dump(saved_weights,f)

        with open("./weights_data/shapes_"+str(prune_rate)+".pkl","wb") as f:
            pkl.dump(saved_shape,f)

        print("[INFO]: weights and shapes have saved.")

    def restore_w(self,sess,weights_file):
        with open(weights_file, "rb") as f:
            weights_data = pkl.load(f)
        for key in weights_data.keys():
            np_w = weights_data[key]
            ts_w = self.__weights[key]
            sess.run(ts_w.assign(np_w))

        print("[INFO]: restore weights finished...")

    def _prune_gradient(self,grads):

        '''apply pruning on gradients

        Args:
            grads: optimizer = tf.train.AdamOptimizer(1e-4)
                   grads = optimizer.compute_gradients(cross_entropy)
                   grads = apply_prune_on_grads(grads)
                   train_op = optimizer.apply_gradients(grads)

        '''
        new_grads = []
        for grad, var in grads:
            for key, w_mask in self.__pruning_mask.items():
                if var.name == key + ":0":
                    w_mask_obj = tf.cast(tf.constant(w_mask), tf.float32)
                    new_grads.append((tf.multiply(w_mask_obj, grad), var))
                    break
        return new_grads

    # network visualization
    def print_w(self,layers_count=None):
        if layers_count is not None:
            t_n = layers_count
        else:
            t_n = self.__total_w_count

        for i in range(t_n):
            ts_w = self.__weights["w_"+str(i)]
            print("layers_"+str(i)+" : "+str(ts_w.get_shape().as_list()))

    def print_network(self,sess):
        for i in sess.graph.get_operations():
            print(i.name)

    def __prune_s_layer_thres(self,sess,sl,threshold):
        '''
            Args:
                sess: tf.Session()
                sl: index of weights
                threshold: pruning threshold

            Return:
                nzero_count, total count

        '''
        ts_weight = self.__weights["w_"+str(sl)]
        np_weight = sess.run(ts_weight)
        nzero_mask = (np.abs(np_weight)>=threshold).astype(np.float32)
        self.__pruning_mask.update({"w_"+str(sl):nzero_mask})

        sess.run(ts_weight.assign(nzero_mask*np_weight))

        return np.sum(nzero_mask),np.prod(ts_weight.shape.as_list())

    def __prune_s_layer_rate(self,sess,sl,rate):
        '''Args are same with __prune_s_layer_thres()'''
        ts_weight = self.__weights["w_"+str(sl)]
        np_weight = sess.run(ts_weight)

        flat_weights = np_weight.flatten().astype(np.float32)
        abs_flat_w = np.abs(flat_weights)
        abs_sorted_w = np.sort(abs_flat_w)

        total_prune = int(flat_weights.shape[0]*rate)

        if total_prune-1<0:
            total_prune=1

        threshold = abs_sorted_w[total_prune-1]

        nzero_mask = (np.abs(np_weight)>=threshold).astype(np.float32)

        self.__pruning_mask.update({"w_"+str(sl):nzero_mask})

        sess.run(ts_weight.assign(nzero_mask*np_weight))

        return np.sum(nzero_mask),np.prod(ts_weight.shape.as_list())

    def _prune_weights(self,sess,prune_type="rate",prune_val=0.05,verbose=True):

        prune_func = None

        if prune_type not in self.__support_prune_weights_type:
            raise ValueError("prune_type must be ",self.__support_prune_weights_type)

        if prune_type=="threshold":
            if prune_val<0.005 or prune_val>1.0:
                raise ValueError("threshold is beyond [0.005, 1.0]")
            prune_func = self.__prune_s_layer_thres
        elif prune_type=="rate":
            if prune_val<=0. or prune_val >=0.95:
                raise ValueError("prune rate is beyond [0., 0.95]")
            prune_func = self.__prune_s_layer_rate

        total_layers = self.__total_w_count
        for sl in range(total_layers):
            nzero_count,total_count = prune_func(sess,sl,prune_val)
            if verbose:
                print("layer name: ",self.__index_remap[sl],
                      " , pruned weights: ",round(1.0-(nzero_count/float(total_count)),4))