import os
import re
import threading
import sys
import copy
import numpy as np
import tensorflow as tf
import glob
DATA_PATH = "../event_based_numpy_merged_all_secs/Jazz"

def generator_fn():
    """for digit in range(4):
        line = 'I am digit {}'.format(digit)
        words = line.split()
        yield [w.encode() for w in words], len(words)"""
    files = glob.glob(DATA_PATH+"/*.npy")
    for f in files:
        yield np.load(f)

if __name__=='__main__':


    #dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=([None], ()), output_types=(tf.string, tf.int32))
    dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=(None), output_types=(tf.float32))

    iterator = dataset.make_one_shot_iterator()
    node = iterator.get_next()
    with tf.Session() as sess:
        for i in range(125):
            data_loaded = sess.run(node)
            print("asdsadsdasd",data_loaded.shape)
        #print("asdasdas",sess.run(node)) 

    """
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    coorder = tf.train.Coordinator()
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess, coord=coorder)
    """