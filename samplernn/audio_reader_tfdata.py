import os
import re
import threading
import sys
import copy
import numpy as np
import tensorflow as tf
import glob

class AudioReader(object):

    def __init__(self, audio_dir,coord, piano_dim, note_channel, rhythm_channel, chord_channel, big_frame_size = 16, seq_len=None, queue_size=16):
        self.audio_dir = audio_dir
        self.piano_dim = piano_dim
        self.note_channel = note_channel
        self.rhythm_channel = rhythm_channel
        self.chord_channel = chord_channel
        self.coord = coord
        self.seq_len = seq_len
        self.big_frame_size = big_frame_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(
            capacity = queue_size, dtypes=[tf.float32], shapes=[(None, self.piano_dim)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def encode_npy(self, mid_files):
        mid_cond = []
        for mid_file in mid_files:
            mid = np.load(mid_file)
            print("aaaaa",mid_file,mid.shape)
            mid_cond.append(mid)
        print("encoding........FINISHED!")
        return mid_cond

    def thread_main2(self, sess):
        stop = False
        mid_files = glob.glob(self.audio_dir+'/*.npy')
        mid_list = self.encode_npy(mid_files)
        while not stop:
            for i,audio_copy in enumerate(mid_list):
                audio = audio_copy
                if self.coord.should_stop():
                    stop = True
                    break

                #pad_elements = self.seq_len - 1 - (audio.shape[0] + self.seq_len - 1) % self.seq_len
                
                #empty = self.find_pad(pad_elements, audio)

                #audio = np.concatenate([audio, empty],axis=0).astype(np.float32)

                while len(audio) >= self.seq_len:
                    piece = audio[:self.seq_len, :]
                    sess.run(self.enqueue,
                                feed_dict={self.sample_placeholder: piece})
                    audio = audio[self.seq_len:, :]
                #sess.run(self.enqueue,feed_dict={self.sample_placeholder: audio}) #should return different len


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main2, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads



    def data_batch(self, batch_size=1, epoch = None):
        """Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            image_paths: A list of paths to individual images
            label_paths: A list of paths to individual label images
            augment: Boolean, whether to augment data or not
            batch_size: Number of images/labels in each batch returned
            num_threads: Number of parallel calls to make
        Returns:
            next_element: A tensor with shape [2], where next_element[0]
                        is image batch, next_element[1] is the corresponding
                        label batch
            init_op: Data initializer op, needs to be executed in a session
                    for the data queue to be filled up and the next_element op
                    to yield batches"""
        def read_npy_file(item):
            audio = np.load(item).astype(np.float32)
            lst = []
            while len(audio) >= self.seq_len:
                piece = audio[:self.seq_len, :]
                lst.append(piece)
                audio = audio[self.seq_len:, :]
            return lst

        mid_files = glob.glob(self.audio_dir+'/*.npy')

        data = tf.data.Dataset.from_tensor_slices(mid_files)

        data = data.map(
                lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,tf.float32, tf.float32])))

        num_threads = 10
        num_prefetch = 5*batch_size
        #data = data.map(_parse_data,
         #           num_parallel_calls=num_threads).prefetch(num_prefetch)


        data = data.shuffle(buffer_size=len(mid_files))

        # Batch, epoch, shuffle the data
        data = data.batch(batch_size)
        data = data.repeat(epoch)

        # Create iterator
        iterator = data.make_one_shot_iterator()

        # Next element Op
        next_element = iterator.get_next()

        return next_element




if __name__=='__main__':
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    coorder = tf.train.Coordinator()
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess, coord=coorder)
    reader = AudioReader(audio_dir = '../event_based_numpy_merged_all_secs_with_bar_pad/Jazz',coord = coorder, piano_dim = 214, note_channel = 131, rhythm_channel = 33, chord_channel =50 ,seq_len=32)
    print('init finish')
    next_ele = reader.data_batch(batch_size = 1)
    nxt_element = sess.run(next_ele)
    for i,f in enumerate(nxt_element):
        print(f.shape)
