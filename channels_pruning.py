#coding:utf-8
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
from modelsets import CifarModelZoo
from datasets import CifarData

from utils.drawCurve import drawLib
from modelBuilder import test_full_model
from utils import configs as cfg

# 文件存放目录
CIFAR_DIR = "./cifar-10-python"

if __name__ == "__main__":

    drawer = drawLib()

    train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data = CifarData(train_filename, True)
    test_data = CifarData(test_filename, True)

    train_batch_size = cfg.TRAIN_BATCHSIZE
    train_batches = train_data._num_examples // train_batch_size
    test_batch_size = cfg.TEST_BATCHSIZE
    test_batches = test_data._num_examples // test_batch_size

    retrain_epoches = cfg.RETRAIN_EPOCH

    test_prune_batches = cfg.TEST_PRUNE_BATCHES #剪枝前的测试组数
    dropout_time_threshold = cfg.DROPOUT_TIME_THRESHOLD #舍弃前多少组的时间

    l2_loss_decay = cfg.L2_LOSS_DECAY

    prune_rate = cfg.PRUNING_RATE
    full_train = cfg.FULL_TRAIN
    model_name = cfg.MODEL_NAME

    # ----------------test full model--------------- #
    test_full_model()
    # -----------------------------------------------#

    # ----------------get pruned wieghts and shapes.--------------- #
    print("[INFO]: start prune channels.")
    input_x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input0")
    input_y = tf.placeholder(tf.int64, [None], name="input1")
    is_train = tf.placeholder(tf.bool, [], name="is_training")

    params = {"inputs":input_x,"is_train":is_train,"reload_w":None,"num_classes":cfg.NUM_CLASSES}
    logits, model = CifarModelZoo.getModel(model_name,params)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./ckpt_model/full_model.ckpt")
        model._prune_channels(sess,"rate",prune_rate)
    print("[INFO]: channels have been pruned.")
    # -------------------------------------------------------------- #

    tf.reset_default_graph()

    # ---------------------reconstruction network---------------------#
    print("[INFO]: start reconstruction network.")
    input_x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input0")
    input_y = tf.placeholder(tf.int64, [None], name="input1")
    is_train = tf.placeholder(tf.bool, [], name="is_training")

    params = {"inputs":input_x,"is_train":is_train,"reload_w":"./weights_data/shapes_"+str(prune_rate)+".pkl","num_classes":cfg.NUM_CLASSES}
    logits, model = CifarModelZoo.getModel(model_name,params)
    softmax_out = tf.nn.softmax(logits, name="softmax")  # sparse_softmax_cross_entropy_with_logits

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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

        if not full_train:
            model.restore_w(sess,"./weights_data/weights_"+str(prune_rate)+".pkl")
        else:
            pritn("fullly train the pruned model.")

        # test before pruning
        all_test_acc_val = []
        t_before = []
        all_test_loss_val = []
        for j in range(test_prune_batches):
            test_batch_data, test_batch_labels = test_data.next_batch(test_batch_size)
            t_start = time.time()
            test_loss_val,test_acc_val = sess.run([loss,accuracy],
                                    feed_dict={input_x: test_batch_data, input_y: test_batch_labels,
                                               is_train: False})
            t_before.append(float(time.time() - t_start) * 1000.)
            all_test_loss_val.append(test_loss_val)
            all_test_acc_val.append(test_acc_val)
        print("[before retrain] , acc:", round(np.mean(all_test_acc_val[dropout_time_threshold:]),4),
              " avg time/ms:", round(np.mean(t_before),4),
              "loss:",round(np.mean(all_test_loss_val),4))

        # start retrain
        for epoch in range(retrain_epoches):
            for i in range(train_batches):
                batch_data, batch_labels = train_data.next_batch(train_batch_size)
                sess.run(train_step,feed_dict={input_x: batch_data, input_y: batch_labels,is_train:True})

                if (i + 1) % cfg.PRINT_TRAIN_INFO_PER_STEP == 0:
                    loss_val,acc_val = sess.run([loss,accuracy],feed_dict={input_x: batch_data, input_y: batch_labels,is_train:False})
                    print('[Retrain] Epoches: %d, Step: %d, loss: %4.5f, acc: %4.5f' % (epoch+1,i + 1, loss_val, acc_val))

                    drawer.drawPts(acc_val,0)
                    drawer.drawPts(loss_val,1,5.0)
                    drawer.update()

                if (i+1)% cfg.TEST_PER_STEP == 0:
                    all_test_acc_val = []
                    all_test_loss_val = []
                    for j in range(test_batches):
                        test_batch_data, test_batch_labels = test_data.next_batch(test_batch_size)
                        test_loss_val,test_acc_val = sess.run([loss,accuracy],
                                                feed_dict={input_x: test_batch_data, input_y: test_batch_labels,is_train:False})
                        all_test_acc_val.append(test_acc_val)
                        all_test_loss_val.append(test_loss_val)
                    test_acc = np.mean(all_test_acc_val)
                    test_loss = np.mean(all_test_loss_val)
                    print('[Test] loss: %4.5f, acc: %4.5f' % (test_loss, test_acc))

                    drawer.drawPts(test_acc,2,5.0)
                    drawer.drawPts(test_loss,3,5.0)
                    drawer.update()

        drawer.save("./logs/channels_pruned.png")

        # test before pruning
        all_test_acc_val = []
        t_before = []
        all_test_loss_val = []
        for j in range(test_prune_batches):
            test_batch_data, test_batch_labels = test_data.next_batch(test_batch_size)
            t_start = time.time()
            test_loss_val,test_acc_val = sess.run([loss,accuracy],
                                    feed_dict={input_x: test_batch_data, input_y: test_batch_labels,
                                               is_train: False})
            t_before.append(float(time.time() - t_start) * 1000.)
            all_test_loss_val.append(test_loss_val)
            all_test_acc_val.append(test_acc_val)
        print("[after retrain] , acc:",  round(np.mean(all_test_acc_val[dropout_time_threshold:]),4),
              " avg time/ms:", round(np.mean(t_before),4),
              "loss:",round(np.mean(all_test_loss_val),4))

        saver.save(sess, "./channels_pruned_model/channels_pruned_model.ckpt")
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
        with tf.gfile.FastGFile("./channels_pruned_model/channels_pruned_model.pb", 'wb') as f:
            f.write(constant_graph.SerializeToString())
        print("model has saved...")

