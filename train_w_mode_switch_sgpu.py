from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import time
from datetime import datetime
import random
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from samplernn import SampleRnnModel_w_mode_switch
from samplernn import AudioReader
from samplernn import mu_law_decode
from samplernn import optimizer_factory
########
LOGDIR_ROOT = './logdir'
DATA_DIRECTORY = 'AUDIO'
COND_DIRECTORY = 'COND'
########
CHECKPOINT_EVERY = 500
########
LEARNING_RATE = 6e-5


SEQ_LEN = 32 
L2_REGULARIZATION_STRENGTH = 0
MOMENTUM = 0.9
MAX_TO_KEEP = 150
PIANO_DIM = 195
NOTE_CHANNEL = 130
RHYTHM_CHANNEL = 16
CHORD_CHANNEL = 49
BAR_CHANNEL = 2
RATIO = 0.1
BATCH_SIZE = 1
NUM_GPU = 1

NUM_STEPS = 120001
def get_arguments():
    parser = argparse.ArgumentParser(description='SampleRnn example network')
    parser.add_argument('--num_gpus',         type=int,   default=NUM_GPU)
    parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
    parser.add_argument('--data_dir',         type=str,
                        default=DATA_DIRECTORY)
    parser.add_argument('--val_data_dir',type=str,)                  
    parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY)
    parser.add_argument('--num_steps',        type=int,   default=NUM_STEPS)
    parser.add_argument('--learning_rate',    type=float,
                        default=LEARNING_RATE)
    parser.add_argument('--l2_regularization_strength',
                        type=float, default=L2_REGULARIZATION_STRENGTH)
    parser.add_argument('--optimizer',        type=str,
                        default='adam', choices=optimizer_factory.keys())
    parser.add_argument('--momentum',         type=float, default=MOMENTUM)

    parser.add_argument('--seq_len',          type=int, default = SEQ_LEN)
    parser.add_argument('--big_frame_size',   type=int, required=True)
    parser.add_argument('--frame_size',       type=int, required=True)
    parser.add_argument('--dim',              type=int, required=True)
    parser.add_argument('--n_rnn',            type=int,
                        choices=list(range(1, 6)), required=True)
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], required=True)
    parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
    parser.add_argument('--saved_path',  type=str, default=None)
    parser.add_argument('--mode_choice', choices=['bar_note', 'nosamplernn','note',"ad_rm2t", "ad_rm3t"], type = str,default='bar_note')
    parser.add_argument('--if_cond',type=str, choices=['cond','no_cond'])
    parser.add_argument('--piano_dim',type=int, default = PIANO_DIM)
    parser.add_argument('--note_channel',type=int, default = NOTE_CHANNEL)
    parser.add_argument('--rhythm_channel',type=int, default = RHYTHM_CHANNEL)
    parser.add_argument('--chord_channel',type=int, default = CHORD_CHANNEL)
    parser.add_argument('--bar_channel',type=int, default = BAR_CHANNEL)
    parser.add_argument('--alpha1',type=float, default = 0.5)
    parser.add_argument('--drop_out',type=float, default = 0.5)   
    parser.add_argument('--alpha2',type=float, default = 0.3) 
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),end="")

    #ckpt = tf.train.get_checkpoint_state(logdir)
    try:
        saver.restore(sess, logdir)
        global_step = int(logdir.split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        print(" Done.")
        return global_step
    except:
        print(" No checkpoint found.")
        return None



def save_gt_pd(gt,pd,step,i,logdir):
    gt_name = logdir+'/'+str(step)+'_'+str(i)+'_gt.npy'
    pd_name = logdir+'/'+str(step)+'_'+str(i)+'_pd.npy'
    np.save(gt_name,gt)
    np.save(pd_name,pd)


def main():
    args = get_arguments()
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    if args.saved_path is not None:
        date_time = args.saved_path.split('/')[-2]+'_contd'
        logdir = os.path.join(args.logdir_root, date_time) 
        if not os.path.exists(logdir):
            os.makedirs(logdir)
     
    else:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        logdir = os.path.join(args.logdir_root, date_time+"_"+args.data_dir.split("/")[-1]+"_"+args.mode_choice)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    ####write the config in####
    with open(logdir+'/config.txt',"w") as f:
        f.write('lr= {}\n'.format(LEARNING_RATE))
        f.write('big frame size: {}\n'.format(args.big_frame_size))
        f.write('frame size: {}\n'.format(args.frame_size))
        f.write('data used: {}\n'.format(args.data_dir))
        f.write('model used: {}\n'.format(args.mode_choice))
        f.write("if cond: {}\n".format(args.if_cond))
        f.write("n_rnn: {}\n".format(args.n_rnn))
        f.write("note_channel: {}\n".format(args.note_channel))
        f.write("rhythm_channel: {}\n".format(args.rhythm_channel))
        f.write("rnn_type: {}\n".format(args.rnn_type))
        f.write("dim: {}\n".format(args.dim))
        f.write('optimizer:{}\n'.format(args.optimizer))
        f.write("chord_channel: {}\n".format(args.chord_channel))    
        f.write("alpha1: {}\n".format(args.alpha1))  
        f.write("dropout: {}\n".format(args.drop_out)) 
        f.write("regularization: {}\n".format(args.l2_regularization_strength)) 
        f.write("bar_channel: {}\n".format(args.bar_channel))
        f.write("alpha2: {}\n".format(args.alpha2))





    ####model config, learning rate, optimizer####
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False
    )
    #lr= tf.train.exponential_decay(args.learning_rate,global_step,8000,0.9,staircase = True)
    lr = args.learning_rate
    optim = optimizer_factory[args.optimizer](learning_rate=lr,momentum=args.momentum)

    ####define graph####
    net = SampleRnnModel_w_mode_switch(if_train = True, args = args)


    ##graph placeholders##
    if args.if_cond == "cond":
        if args.mode_choice=="nosamplernn":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.frame_size, args.piano_dim), name = "input_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, 1, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
        elif args.mode_choice=="bar_note":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim), name = "input_batch_rnn")
            #network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size-args.big_frame_size, args.piano_dim), name = "output_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.big_frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
        elif args.mode_choice=="note":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim), name = "input_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
        elif args.mode_choice=="ad_rm2t":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim), name = "input_batch_rnn")
            rm_time_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.rhythm_channel), name = "rm_tm_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
    else:
        if args.mode_choice=="nosamplernn":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.frame_size, args.piano_dim-args.chord_channel), name = "input_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, 1, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
        elif args.mode_choice=="bar_note":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim-args.chord_channel), name = "input_batch_rnn")
            #network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size-args.big_frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.big_frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")

        elif args.mode_choice=="note":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim-args.chord_channel), name = "input_batch_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")
        elif args.mode_choice=="ad_rm2t":
            network_input_plder= tf.placeholder(tf.float32,shape =(None, args.seq_len, args.piano_dim-args.chord_channel), name = "input_batch_rnn")
            rm_time_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.rhythm_channel), name = "rm_tm_rnn")
            network_output_plder = tf.placeholder(tf.float32,shape =(None, args.seq_len-args.frame_size, args.piano_dim-args.chord_channel), name = "output_batch_rnn")

    ##build graph##
    with tf.variable_scope(tf.get_variable_scope(),reuse = tf.AUTO_REUSE):
        with tf.name_scope('TOWER_0') as scope:

            if args.mode_choice!="ad_rm2t":
                (   gt,
                    pd,
                    loss
                )=net.loss_SampleRnn(
                    X = network_input_plder,
                    y = network_output_plder,
                    l2_regularization_strength=args.l2_regularization_strength  # noqa: E501
                )
            else:
                (   gt,
                    pd,
                    loss
                )=net.loss_SampleRnn(
                    X = network_input_plder,
                    y = network_output_plder,
                    rm_time= rm_time_plder,
                    l2_regularization_strength=args.l2_regularization_strength  # noqa: E501
                )                
            tf.get_variable_scope().reuse_variables()
            trainable = tf.trainable_variables()
            gradients_vars = optim.compute_gradients(
                loss,
                trainable,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N  # noqa: E501
            )

            gradients, variables = zip(*gradients_vars)
            gradients_clipped, _ = tf.clip_by_global_norm(gradients, 5)

            gradients_vars_clipped = zip(gradients_clipped, variables)

    apply_gradient_op = optim.apply_gradients(gradients_vars, global_step=global_step)

    ####summary####
    writer = tf.summary.FileWriter(logdir+"/train")
    writer2 = tf.summary.FileWriter(logdir+"/test")
    writer.add_graph(tf.get_default_graph())

    tf.summary.scalar('learning_rate',lr)
    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()

    ####session config####
    tf_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=MAX_TO_KEEP)

    try:
        saved_global_step = load(saver, sess, args.saved_path)
        if saved_global_step is None:
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = None
    last_saved_step = saved_global_step


    ####read data and condition####

    with tf.name_scope('create_inputs'):
        reader = AudioReader(coord =coord,
                             args = args)

        audio_batch = reader.dequeue(args.batch_size)
    reader.start_threads(sess)
    ####forward prop and gradient descent####
    try:
        if args.mode_choice!="ad_rm2t":
            X_val, y_val = reader.get_validation_data()
            print("fetched val data sucessfully", X_val.shape, y_val.shape)
            for step in range(saved_global_step + 1, args.num_steps):
                start_time = time.time()
                X, y = sess.run(audio_batch) #[ [(batch,seq,dim)]     [(batch,seq,dim)] ]
                ## define output list ##
                loss_sum_train = 0
                loss_sum_test = 0
                idx_begin = 0

                outp_list_train = [gt,pd, summaries, loss, apply_gradient_op]
                outp_list_test = [gt,pd,summaries, loss]
                ## define input dict (placeholder: input) ##

                inp_dict_train = {}
                inp_dict_test = {}

                ## you can change here! if you want a really long seq with minimal padding
                inp_dict_train[network_input_plder] = X
                inp_dict_train[network_output_plder] = y

                inp_dict_test[network_input_plder] = X_val
                inp_dict_test[network_output_plder] = y_val
                ## run train op ##
                ground_truth_train, prediction_train, summary_train, loss_train, _ = \
                    sess.run(outp_list_train, feed_dict=inp_dict_train) #inp_dict(audio_pld, condition_pld, train_big, train_state)
                ## run test op ##
                if step % 500 == 0:
                    ground_truth_test, prediction_test, summary_test, loss_test = \
                        sess.run(outp_list_test, feed_dict=inp_dict_test) #inp_dict(audio_pld, condition_pld, train_big, train_state)
                    writer2.add_summary(summary_test, step)
                #if step % 300 == 0:
                    #save_gt_pd(ground_truth,prediction,step,i,logdir)
                writer.add_summary(summary_train, step)
                
                duration = time.time() - start_time
                print('step {:d} - train_loss = {:.3f}, test_loss = {:.3f}, ({:.3f} sec/step)'
                    .format(step, loss_train, loss_test, duration))
                
                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    last_saved_step = step
        else:
            X_val, y_val, remaining_time_val = reader.get_adrm_validation_data()
            print("fetched val data sucessfully", X_val.shape, y_val.shape,remaining_time_val.shape)
            for step in range(saved_global_step + 1, args.num_steps):
                start_time = time.time()
                X, y, remaining_time = sess.run(audio_batch) #[ [(batch,seq,dim)]     [(batch,seq,dim)] ]
                ## define output list ##
                loss_sum_train = 0
                loss_sum_test = 0
                idx_begin = 0

                outp_list_train = [gt,pd, summaries, loss, apply_gradient_op]
                outp_list_test = [gt,pd,summaries, loss]
                ## define input dict (placeholder: input) ##

                inp_dict_train = {}
                inp_dict_test = {}

                ## you can change here! if you want a really long seq with minimal padding
                inp_dict_train[network_input_plder] = X
                inp_dict_train[network_output_plder] = y
                inp_dict_train[rm_time_plder] = remaining_time 

                inp_dict_test[network_input_plder] = X_val
                inp_dict_test[network_output_plder] = y_val
                inp_dict_test[rm_time_plder] = remaining_time_val 
                ## run train op ##
                ground_truth_train, prediction_train, summary_train, loss_train, _ = \
                    sess.run(outp_list_train, feed_dict=inp_dict_train) #inp_dict(audio_pld, condition_pld, train_big, train_state)
                ## run test op ##
                if step % 500 == 0:
                    ground_truth_test, prediction_test, summary_test, loss_test = \
                        sess.run(outp_list_test, feed_dict=inp_dict_test) #inp_dict(audio_pld, condition_pld, train_big, train_state)
                    writer2.add_summary(summary_test, step)
                #if step % 300 == 0:
                    #save_gt_pd(ground_truth,prediction,step,i,logdir)
                writer.add_summary(summary_train, step)
                
                duration = time.time() - start_time
                print('step {:d} - train_loss = {:.3f}, test_loss = {:.3f}, ({:.3f} sec/step)'
                    .format(step, loss_train, loss_test, duration))
                
                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    last_saved_step = step
    except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
