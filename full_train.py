#coding:utf-8

'''
author:Wang Haibo
at: Pingan Tec.
email: haibo.david@qq.com

!!!
代码中会有少量中文注释，无需在意

'''
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from modelsets import CifarModelZoo
from datasets import CifarData

from utils.drawCurve import drawLib
from utils import configs as cfg

import time

if __name__ == "__main__":

    drawer = drawLib()

    CIFAR_DIR = "./cifar-10-python"
    train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]

    train_data = CifarData(train_filename, True)
    test_data = CifarData(test_filename, True)

    train_batch_size = cfg.TRAIN_BATCHSIZE
    train_batches = train_data._num_examples // train_batch_size
    test_batch_size = cfg.TEST_BATCHSIZE
    test_batches = test_data._num_examples // test_batch_size

    train_epoches = cfg.TRAIN_EPOCHES

    restore_model = cfg.RESTORE_MODEL

    l2_loss_decay = cfg.L2_LOSS_DECAY

    input_x = tf.placeholder(tf.float32, [None,32,32,3],name="input0")
    input_y = tf.placeholder(tf.int64, [None],name="input1")
    is_train = tf.placeholder(tf.bool,[],name="is_training")

    params = {"inputs":input_x,"is_train":is_train,"reload_w":None,"num_classes":cfg.NUM_CLASSES}

    logits, model = CifarModelZoo.getModel(cfg.MODEL_NAME,params)

    saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    softmax_out = tf.nn.softmax(logits, name="softmax")
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=input_y))

    l2_loss = l2_loss_decay*tf.add_n([tf.nn.l2_loss(tf.cast(v,tf.float32))for v in tf.trainable_variables()])

    loss += l2_loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(cfg.INIT_LR)
        capped_gvs = optimizer.compute_gradients(loss)
        # capped_gvs = [(tf.clip_by_value(grad, -5e+3, 5e+3), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs)

    correct_prediction = tf.equal(tf.argmax(softmax_out, 1), input_y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if restore_model:
            saver.restore(sess,"./ckpt_model/full_model.ckpt")
            print("[INFO] full_model.ckpt has been restored.")

        for epoch in range(train_epoches):
            for i in range(train_batches):
                batch_data, batch_labels = train_data.next_batch(train_batch_size)
                start_time = time.time()
                sess.run(train_step,feed_dict={input_x: batch_data, input_y: batch_labels,is_train:True})
                step_time = time.time()-start_time
                if (i + 1) % cfg.PRINT_TRAIN_INFO_PER_STEP == 0:
                    loss_val,acc_val = sess.run([loss,accuracy],feed_dict={input_x: batch_data, input_y: batch_labels,is_train:False})
                    print('[Train] Epoch: %d, Step: %d, loss: %4.5f, acc: %4.5f, time:%4.3fms/sample'
                          % (epoch+1,i + 1, loss_val, acc_val, (step_time*1000.)/(cfg.PRINT_TRAIN_INFO_PER_STEP*cfg.TRAIN_BATCHSIZE)))

                    drawer.drawPts(acc_val,0)
                    drawer.drawPts(loss_val,1,5.0)
                    drawer.update()

                if (i+1)% cfg.TEST_PER_STEP == 0:
                    all_test_acc_val = []
                    for j in range(test_batches):
                        test_batch_data, test_batch_labels = test_data.next_batch(test_batch_size)
                        test_acc_val = sess.run([accuracy],
                                                feed_dict={input_x: test_batch_data, input_y: test_batch_labels,is_train:False})
                        all_test_acc_val.append(test_acc_val)
                    test_acc = np.mean(all_test_acc_val)
                    print('[Test] Step: %d, acc: %4.5f' % ((i + 1), test_acc))

                    drawer.drawPts(test_acc, 2)
                    drawer.update()

                    saver.save(sess, "./ckpt_model/full_model.ckpt")
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
                    with tf.gfile.FastGFile("./ckpt_model/full_model.pb", 'wb') as f:
                        f.write(constant_graph.SerializeToString())
                    print("model has saved...")

        saver.save(sess, "./ckpt_model/full_model.ckpt")
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
        with tf.gfile.FastGFile("./ckpt_model/full_model.pb", 'wb') as f:
            f.write(constant_graph.SerializeToString())
        print("model has saved...")
        drawer.save("./logs/train_process.png")

