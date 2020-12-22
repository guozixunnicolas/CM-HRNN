import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
import os
import numpy as np


class SampleRnnModel_w_mode_switch(object):
    def __init__(self, args, if_train):
        self.batch_size = args.batch_size
        self.big_frame_size = args.big_frame_size
        self.frame_size = args.frame_size

        self.rnn_type = args.rnn_type
        self.dim = args.dim
        self.birnndim = args.birnndim
        self.piano_dim = args.piano_dim
        self.n_rnn = args.n_rnn
        self.seq_len = args.seq_len
        self.mode_choice = args.mode_choice
        self.note_channel = args.note_channel
        self.rhythm_channel = args.rhythm_channel
        self.chord_channel = args.chord_channel
        self.bar_channel = args.bar_channel
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.if_train = if_train
        self.drop_out_keep_prob = args.drop_out
        if args.if_cond =="cond":
            self.if_cond = True
            print("model setting: conditional")
        elif args.if_cond=="no_cond":
            self.if_cond = False
            print("model setting: unconditional")

        def single_cell(if_attention = False, atten_len = None):
            if self.rnn_type =="GRU":
                cell = tf.contrib.rnn.GRUCell(self.dim)
                #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=2)
                if if_attention == True:
                    print("with fucking attention")
                    cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=atten_len)
                if self.if_train:
                    cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.drop_out_keep_prob)
                return cell
            elif self.rnn_type =="LSTM":
                print("yeah lstm")
                cell = tf.contrib.rnn.BasicLSTMCell(self.dim)
                if if_attention == True:
                    print("with fucking attention")
                    cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=atten_len)

                if self.if_train:
                    cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.drop_out_keep_prob)
                return cell
        def single_birnncell():
            if self.rnn_type =="GRU":
                cell = tf.contrib.rnn.GRUCell(self.birnndim)
                if self.if_train:
                    cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.drop_out_keep_prob)

                return cell
            elif self.rnn_type =="LSTM":
                cell = tf.contrib.rnn.BasicLSTMCell(self.birnndim)
                if self.if_train:
                    cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.drop_out_keep_prob)

                return cell
        if self.n_rnn > 1:
            self.sample_cell = tf.contrib.rnn.MultiRNNCell([single_cell(if_attention = True, atten_len = self.frame_size) for _ in range(self.n_rnn)])
            self.frame_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.n_rnn)])
            self.big_frame_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.n_rnn)])
            self.birnn_fwcell = single_birnncell()
            self.birnn_bwcell = single_birnncell()
        else:
            self.sample_cell = single_cell(if_attention = True, atten_len = self.frame_size)
            self.frame_cell = single_cell()
            self.big_frame_cell = single_cell()
            self.birnn_fwcell = single_birnncell()
            self.birnn_bwcell = single_birnncell()
        self.sample_init = self.sample_cell.zero_state(self.batch_size, tf.float32)
        self.frame_init = self.frame_cell.zero_state(self.batch_size, tf.float32)
        self.big_frame_init = self.big_frame_cell.zero_state(self.batch_size, tf.float32) 
        self.birnn_fw_init = self.birnn_fwcell.zero_state(self.batch_size, tf.float32)
        self.birnn_bw_init = self.birnn_bwcell.zero_state(self.batch_size, tf.float32)
        #self.cell = single_cell()
        #self.big_cell = single_cell()      
        #self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        #self.big_initial_state = self.big_cell.zero_state(self.batch_size, tf.float32)

    def weight_bias(self,tensor_in,dim,name):
        with tf.variable_scope(name):
            W_initializer =tf.contrib.layers.xavier_initializer()
            b_initializer = tf.constant_initializer()
            W = tf.get_variable(name = name+'_W', shape=(tensor_in.shape[-1],dim), initializer=W_initializer, dtype=tf.float32, trainable=True)
            b = tf.get_variable(name = name+'_b', shape=(dim), initializer=b_initializer, dtype=tf.float32, trainable=True)
            out = tf.add(tf.matmul(tensor_in, W), b)
            #print("tensor in:{}, W:{}, b:{}, out:{}".format(tensor_in.shape, W.shape, b.shape, out.shape))
        return out

    
    def big_frame_level(self, big_frame_input, big_frame_state = None):
 
        big_frame_input_chunks = tf.reshape(big_frame_input,[-1, #batch
                                                                    int(big_frame_input.shape[1]) // self.big_frame_size, #no_of_chunks 
                                                                     self.big_frame_size*int(big_frame_input.shape[-1])]) #frame_size*merged_dim
        with tf.variable_scope("BIG_FRAME_RNN"):
            if big_frame_state is not None: #during generation
                big_frame_outputs_all_stps, big_frame_last_state = tf.nn.dynamic_rnn(self.big_frame_cell, big_frame_input_chunks,initial_state = big_frame_state, dtype=tf.float32)

            else: #during training
                big_frame_outputs_all_stps, big_frame_last_state = tf.nn.dynamic_rnn(self.big_frame_cell, big_frame_input_chunks, dtype=tf.float32) #batch, no_chunks, dim

            big_frame_outputs_all_upsample = self.weight_bias(big_frame_outputs_all_stps, self.dim*self.big_frame_size//self.frame_size, 'emb_big_frame_proj') #batch, no_chunks, dim*big_size/small_size
            big_frame_outputs = tf.reshape(big_frame_outputs_all_upsample,
                                       [tf.shape(big_frame_outputs_all_upsample)[0],
                                        tf.shape(big_frame_outputs_all_upsample)[1] * self.big_frame_size//self.frame_size,
                                        self.dim]) #(batch, no_frame_chunks*ratio, dim)

            return big_frame_outputs, big_frame_last_state            


    def frame_level(self, frame_input, frame_state = None ,bigframe_output = None):
        frame_input_chunks = tf.reshape(frame_input,[-1, #batch
                                                    int(frame_input.shape[1]) // self.frame_size, #no_of_chunks 
                                                    self.frame_size*int(frame_input.shape[-1])]) #frame_size*merged_dim
        print("input chunk shape",frame_input_chunks)
        with tf.variable_scope("FRAME_RNN"):
            
            """if bigframe_output is not None:
                frame_input_chunks = self.weight_bias(frame_input_chunks, self.dim, 'emb_frame_chunks')
                frame_input_chunks += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim """        
            
            
            if frame_state is not None: #during generation
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks,initial_state = frame_state, dtype=tf.float32)
            else: #during training
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks, dtype=tf.float32)
            if bigframe_output is not None:
                frame_outputs_all_stps += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim
            frame_outputs_all_upsample = self.weight_bias(frame_outputs_all_stps, self.dim*self.frame_size, 'emb_frame_proj')
            frame_outputs = tf.reshape(frame_outputs_all_upsample,
                                       [tf.shape(frame_outputs_all_upsample)[0],
                                        tf.shape(frame_outputs_all_upsample)[1] * self.frame_size,
                                        self.dim]) #(batch, n_frame*frame_size, dim)
            return frame_outputs, frame_last_state    

    def frame_level_switch(self, frame_input, frame_state = None ,bigframe_output = None, if_rs = True):
        frame_input_chunks = tf.reshape(frame_input,[-1, #batch
                                                    int(frame_input.shape[1]) // self.frame_size, #no_of_chunks 
                                                    self.frame_size*int(frame_input.shape[-1])]) #frame_size*merged_dim
        with tf.variable_scope("FRAME_RNN"):
            
            if bigframe_output is not None:
                frame_input_chunks = self.weight_bias(frame_input_chunks, self.dim, 'emb_frame_chunks')
                frame_input_chunks += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim      
            
            if frame_state is not None: #during generation
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks,initial_state = frame_state, dtype=tf.float32)
            else: #during training
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks, dtype=tf.float32)
            if bigframe_output is not None and if_rs is True:
                print("hey, residual connection!")
                frame_outputs_all_stps += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim
            frame_outputs_all_upsample = self.weight_bias(frame_outputs_all_stps, self.dim*self.frame_size, 'emb_frame_proj')
            frame_outputs = tf.reshape(frame_outputs_all_upsample,
                                       [tf.shape(frame_outputs_all_upsample)[0],
                                        tf.shape(frame_outputs_all_upsample)[1] * self.frame_size,
                                        self.dim]) #(batch, n_frame*frame_size, dim)
            return frame_outputs, frame_last_state                
       
        
    def frame_level_before(self, frame_input, frame_state = None ,bigframe_output = None):
        frame_input_chunks = tf.reshape(frame_input,[-1, #batch
                                                    int(frame_input.shape[1]) // self.frame_size, #no_of_chunks 
                                                    self.frame_size*int(frame_input.shape[-1])]) #frame_size*merged_dim
        with tf.variable_scope("FRAME_RNN"):
            
            if bigframe_output is not None:
                frame_input_chunks = self.weight_bias(frame_input_chunks, self.dim, 'emb_frame_chunks')
                frame_input_chunks += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim      
            
            if frame_state is not None: #during generation
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks,initial_state = frame_state, dtype=tf.float32)
            else: #during training
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks, dtype=tf.float32)
            #if bigframe_output is not None:
            #    frame_outputs_all_stps += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim
            frame_outputs_all_upsample = self.weight_bias(frame_outputs_all_stps, self.dim*self.frame_size, 'emb_frame_proj')
            frame_outputs = tf.reshape(frame_outputs_all_upsample,
                                       [tf.shape(frame_outputs_all_upsample)[0],
                                        tf.shape(frame_outputs_all_upsample)[1] * self.frame_size,
                                        self.dim]) #(batch, n_frame*frame_size, dim)
            return frame_outputs, frame_last_state                

    def frame_level_residual(self, frame_input, frame_state = None ,bigframe_output = None):
        frame_input_chunks = tf.reshape(frame_input,[-1, #batch
                                                    int(frame_input.shape[1]) // self.frame_size, #no_of_chunks 
                                                    self.frame_size*int(frame_input.shape[-1])]) #frame_size*merged_dim
        with tf.variable_scope("FRAME_RNN"):
            
            if bigframe_output is not None:
                frame_input_chunks = self.weight_bias(frame_input_chunks, self.dim, 'emb_frame_chunks')
                frame_input_chunks += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim      
            
            if frame_state is not None: #during generation
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks,initial_state = frame_state, dtype=tf.float32)
            else: #during training
                frame_outputs_all_stps, frame_last_state = tf.nn.dynamic_rnn(self.frame_cell, frame_input_chunks, dtype=tf.float32)
            if bigframe_output is not None:
                frame_outputs_all_stps += bigframe_output #batch, no_chunk, dim + batch, no_chunk, dim
            frame_outputs_all_upsample = self.weight_bias(frame_outputs_all_stps, self.dim*self.frame_size, 'emb_frame_proj')
            frame_outputs = tf.reshape(frame_outputs_all_upsample,
                                       [tf.shape(frame_outputs_all_upsample)[0],
                                        tf.shape(frame_outputs_all_upsample)[1] * self.frame_size,
                                        self.dim]) #(batch, n_frame*frame_size, dim)
            return frame_outputs, frame_last_state  
    def sample_level(self, sample_input_sequences, frame_output = None, rm_time=None):
        
        sample_filter_shape = [self.frame_size, sample_input_sequences.shape[-1], self.dim]
        sample_filter = tf.get_variable(
            "sample_filter",
            sample_filter_shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        mlp_out = tf.nn.conv1d(sample_input_sequences,
                            sample_filter,
                            stride=1,
                            padding="VALID",
                            name="sample_conv") #(batch, seqlen-framesize, dim)
        if frame_output is not None:
            logits = mlp_out+frame_output
        if rm_time is not None:
            logits = tf.concat([logits, rm_time],axis = -1)

        

        """sample_filter_shape = [self.frame_size, sample_input_sequences.shape[-1], sample_input_sequences.shape[-1]]
        ones= tf.ones(sample_filter_shape,dtype="float32")     
        same_filter= tf.diag(ones)
        print("same filter",same_filter.shape)
        mlp_out = tf.nn.conv1d(sample_input_sequences,
                            same_filter,
                            stride=1,
                            padding="VALID",
                            name="same_conv") #(batch, seqlen-framesize, dim)
        print("mlp_out shape",mlp_out.shape,sample_input_sequences)
        sample_outputs_all, sample_state_new = tf.nn.dynamic_rnn(self.sample_cell, mlp_out,
                                   dtype=tf.float32)        
        if frame_output is not None:
            logits = sample_outputs_all+frame_output"""


        sample_outputs = self.weight_bias(logits, self.piano_dim-self.chord_channel ,"dense_weights_0") 
        sample_outputs = tf.nn.relu(sample_outputs)
        sample_outputs = self.weight_bias(logits, self.piano_dim-self.chord_channel ,"dense_weights_1") 
        sample_outputs = tf.nn.relu(sample_outputs)
        sample_outputs_logits = self.weight_bias(logits, self.piano_dim-self.chord_channel ,"dense_weights_2")             
        
        return sample_outputs_logits  #return (batch, pred_length, piano_dim)

    def bln_attn(self, baseline_input, baseline_state = None):
        with tf.variable_scope("baseline"):
            if baseline_state is not None: #during generation
                bln_outputs_all, baseline_last_state = tf.nn.dynamic_rnn(self.sample_cell, baseline_input,initial_state = baseline_state, dtype=tf.float32)

            else: #during training
                bln_outputs_all, baseline_last_state = tf.nn.dynamic_rnn(self.sample_cell, baseline_input, dtype=tf.float32) #batch, no_chunks, dim

            baseline_outputs_all_stps = self.weight_bias(bln_outputs_all, self.piano_dim-self.chord_channel ,"dense_weights_bln")
        return baseline_outputs_all_stps, baseline_last_state

    def birnn(self, all_chords):
        
        outputs_tuple,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.birnn_fwcell, cell_bw = self.birnn_bwcell,inputs = all_chords, dtype=tf.float32)

        #outputs_tuple,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell, cell_bw = backward_cell,inputs = all_chords, dtype=tf.float32)
        outputs = tf.concat([outputs_tuple[0],outputs_tuple[1]], axis = -1)
        #all_chords_birnn = self.weight_bias(outputs, 2*self.chord_channel ,"birnn_weights_3") 
        all_chords_birnn = self.weight_bias(outputs, self.chord_channel ,"birnn_weights_3") 
        return all_chords_birnn
    def _create_network_two_layer_SampleRnn(self, two_t_input):
        print("####MODEL:NOTE BAR...####")
        with tf.name_scope('SampleRnn_net'):
            #sample_input = two_t_input[:,self.big_frame_size:-1,:]
    
            #frame_input = two_t_input[:, self.big_frame_size:,:] 

            sample_input = two_t_input[:,self.big_frame_size-self.frame_size:-1,:]

            frame_input = two_t_input[:, self.big_frame_size-self.frame_size:-self.frame_size,:] 

            big_frame_input = two_t_input[:,:-self.big_frame_size,:]  

            big_frame_outputs , final_big_frame_state = self.big_frame_level(big_frame_input)

            frame_outputs , final_frame_state = self.frame_level(frame_input, bigframe_output = big_frame_outputs)

            ##sample_level## 
            sample_logits= self.sample_level(sample_input, frame_output = frame_outputs)

            return sample_logits


    def _create_network_one_layer_SampleRnn(self, one_t_input):
        print("####MODEL:BAR...####")
        sample_input = one_t_input[:,:-1,:]
 
        frame_input = one_t_input[:, :-self.frame_size,:] 

        ##frame_level##
        frame_outputs , final_frame_state = self.frame_level(frame_input)
        ##sample_level## 
        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs)
        return sample_logits

    def _create_network_2t_fc(self, one_t_input):
        print("####MODEL:BAR...####")
        sample_input = one_t_input[:,:-1,:]
 
        frame_input = one_t_input[:, :-self.frame_size,:] 

        ##frame_level##
        frame_outputs , final_frame_state = self.frame_level(frame_input)
        ##sample_level## 
        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs)
        return sample_logits

    def _create_network_noSampleRnn(self, baseline_input):

        sample_outputs_all, sample_state_new = tf.nn.dynamic_rnn(self.sample_cell, baseline_input,
                                   dtype=tf.float32)
        sample_outputs_last = sample_outputs_all[:, -1, :] # (iter*batch, dim)
        #sample_outputs_logits = self.weight_bias(sample_outputs_last, self.note_channel,"dense_weights_3")
        sample_outputs_logits = self.weight_bias(sample_outputs_last, self.piano_dim-self.chord_channel ,"dense_weights_3") 
        sample_outputs_logits = tf.nn.relu(sample_outputs_logits)# (iter*batch, note_channel+rhythm)
        return sample_outputs_logits

    def _create_network_ad_rm2t(self, one_t_input, rm_tm):
        sample_input = one_t_input[:,:-1,:] # batch, seq-1, piano_dim
 
        frame_input = one_t_input[:, :-self.frame_size,:] #(batch, seq-frame_size, piano_dim)
        remaining_time_input = rm_tm #(batch, seq-frame_size, piano_dim)
        print("fram input dim",frame_input)
        ##frame_level##
        frame_outputs , final_frame_state = self.frame_level(frame_input)
        ##sample_level## 
        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs, rm_time = remaining_time_input)
        return sample_logits        
    def _create_network_ad_rm3t(self, two_t_input, rm_tm):
        print("yes ad rm 3t")
        #big frame level
        big_frame_input = two_t_input[:,:-self.big_frame_size,:]  

        big_frame_outputs , final_big_frame_state = self.big_frame_level(big_frame_input)

        #frame level
        frame_input = two_t_input[:, self.big_frame_size-self.frame_size:-self.frame_size,:]

        frame_outputs , final_frame_state = self.frame_level(frame_input, bigframe_output = big_frame_outputs)

        ##sample level
        remaining_time_input = rm_tm

        sample_input = two_t_input[:,self.big_frame_size-self.frame_size:-1,:]

        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs, rm_time = remaining_time_input)

        return sample_logits    

    def _create_network_ad_rm2t_birnn(self, one_t_input, rm_tm):
        #get bidirectional rnn

        #seperate input
        all_bars = one_t_input[:,:,:self.bar_channel]
        all_chords = one_t_input[:,:,self.bar_channel:self.bar_channel+self.chord_channel]
        all_rhythms_melodies = one_t_input[:,:,self.bar_channel+self.chord_channel:]


        #all_chords_birnn = self.birnn(self.birnn_fwcell,self.birnn_bwcell,all_chords)
        all_chords_birnn = self.birnn(all_chords)
        one_t_input_new = tf.concat([all_bars, all_chords_birnn, all_rhythms_melodies], axis = -1)
        sample_input = one_t_input_new[:,:-1,:] # batch, seq-1, piano_dim
 
        frame_input = one_t_input_new[:, :-self.frame_size,:] #(batch, seq-frame_size, piano_dim)
        remaining_time_input = rm_tm #(batch, seq-frame_size, piano_dim)
        ##frame_level##
        frame_outputs , final_frame_state = self.frame_level(frame_input)
        ##sample_level## 
        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs, rm_time = remaining_time_input)
        return sample_logits  

    def _create_network_ad_rm2t_fc(self, one_t_input, rm_tm):
        sample_input = one_t_input[:,:-1,:] # batch, seq-1, piano_dim
 
        frame_input = one_t_input[:, :-self.frame_size,:] #(batch, seq-frame_size, piano_dim)
        remaining_time_input = rm_tm #(batch, seq-frame_size, piano_dim)
        print("fram input dim",frame_input)
        ##frame_level##
        frame_outputs , final_frame_state = self.frame_level(frame_input)
        ##sample_level## 
        sample_logits= self.sample_level(sample_input, frame_output = frame_outputs, rm_time = remaining_time_input)
        return sample_logits        

    def _create_network_ad_rm3t_fc(self, two_t_input,rm_tm, if_rs = True):
        print("_create_network_ad_rm3t_fc")
        with tf.name_scope('SampleRnn_net'):

            sample_input = two_t_input[:,self.big_frame_size-self.frame_size:-1,:]

            frame_input = two_t_input[:, self.big_frame_size-self.frame_size:-self.frame_size,:] 

            big_frame_input = two_t_input[:,:-self.big_frame_size,:]  

            big_frame_outputs , final_big_frame_state = self.big_frame_level(big_frame_input)

            frame_outputs , final_frame_state = self.frame_level_switch(frame_input, bigframe_output = big_frame_outputs, if_rs = if_rs)
            #frame_outputs , final_frame_state = self.frame_level(frame_input, bigframe_output = big_frame_outputs)

            remaining_time_input = rm_tm #(batch, seq-frame_size, piano_dim)
            ##sample_level## 
            sample_logits= self.sample_level(sample_input, frame_output = frame_outputs, rm_time = remaining_time_input)

            return sample_logits
    def _create_network_bln_attn_fc(self, baseline_input):
        bln_outputs_logits,_ = self.bln_attn(baseline_input)
        return bln_outputs_logits
    def loss_SampleRnn(self, X,y, rm_time = None,l2_regularization_strength=None, name='sample'):
        """ nosamplernn: X: (batch, frame_size, dim), Y:(batch, 1, dim)
            barnote: X(batch, seq_len, dim), Y:(batch, seq_len-frame-big_frame, dim)
            note: X(batch, seq_len, dim), Y:(batch, seq_len-frame, dim)

            #dim here is 211 for cond, 211-49 for no_cond
        """
        self.X = X
        self.y = y
        self.rm_time = rm_time

        with tf.name_scope(name):
            if self.mode_choice =="bar_note":
                pred_logits = self._create_network_two_layer_SampleRnn(two_t_input = self.X) #(batch* seq_len-frame-big_frame, self.note + self.rhythm)
                pd = pred_logits

            elif self.mode_choice =="note":
                pred_logits= self._create_network_one_layer_SampleRnn(one_t_input = self.X) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits
            elif self.mode_choice=="nosamplernn":
                pred_logits = self._create_network_noSampleRnn( baseline_input= self.X) #(batch, self.note + self.rhythm)
                pd = tf.expand_dims(pred_logits, axis = 1) #(batch, 1, self.note + self.rhythm)
            elif self.mode_choice=="ad_rm2t":
                pred_logits= self._create_network_ad_rm2t(one_t_input = self.X, rm_tm = self.rm_time) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits       
            elif self.mode_choice=="ad_rm3t":  
                pred_logits= self._create_network_ad_rm3t(two_t_input = self.X, rm_tm = self.rm_time) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits  
            elif self.mode_choice=="ad_rm2t_birnn":  
                pred_logits= self._create_network_ad_rm2t_birnn(one_t_input = self.X, rm_tm = self.rm_time) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits     
            elif self.mode_choice=="ad_rm2t_fc":  
                pred_logits= self._create_network_ad_rm2t_fc(one_t_input = self.X, rm_tm = self.rm_time) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits 
            elif self.mode_choice=="ad_rm3t_fc":  
                pred_logits= self._create_network_ad_rm3t_fc(two_t_input = self.X, rm_tm = self.rm_time, if_rs = False) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits                 
            elif self.mode_choice=="ad_rm3t_fc_rs":  
                pred_logits= self._create_network_ad_rm3t_fc(two_t_input = self.X, rm_tm = self.rm_time, if_rs = True) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits    
            elif self.mode_choice=="bln_attn_fc":  
                pred_logits= self._create_network_bln_attn_fc(baseline_input = self.X) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits       
            elif self.mode_choice=="2t_fc":  
                pred_logits= self._create_network_2t_fc(one_t_input = self.X) #(batch* seq_len-frame, self.note + self.rhythm)
                pd = pred_logits                                   
            gt_bar = self.y[:, :, :self.bar_channel]
            gt_sustain = self.y[:, :, self.bar_channel : self.bar_channel+self.rhythm_channel]
            gt_note = self.y[:,:, self.bar_channel+self.rhythm_channel:]

            pd_bar = pd[:, :, :self.bar_channel]
            pd_note = pd[:,:, self.bar_channel+self.rhythm_channel:]
            pd_sustain = pd[:, :, self.bar_channel : self.bar_channel+self.rhythm_channel]

            gt_note = tf.reshape(gt_note, [-1, self.note_channel])
            pd_note = tf.reshape(pd_note, [-1, self.note_channel])

            gt_sustain = tf.reshape(gt_sustain, [-1, self.rhythm_channel])
            pd_sustain = tf.reshape(pd_sustain, [-1, self.rhythm_channel])

            gt_bar = tf.reshape(gt_bar, [-1, self.bar_channel])
            pd_bar = tf.reshape(pd_bar, [-1, self.bar_channel])

            with tf.name_scope('sample_RNN_loss'):
                loss_note = tf.nn.softmax_cross_entropy_with_logits(logits=pd_note, labels=gt_note)
                loss_rhythm = tf.nn.softmax_cross_entropy_with_logits(logits=pd_sustain, labels=gt_sustain)
                loss_bar = tf.nn.softmax_cross_entropy_with_logits(logits=pd_bar, labels=gt_bar)
                loss = self.alpha1 * loss_note + self.alpha2*loss_rhythm + (1-self.alpha1-self.alpha2)*loss_bar

                reduced_loss = tf.reduce_mean(loss)
                reduced_note_loss = tf.reduce_mean(loss_note)
                reduced_rhythm_loss = tf.reduce_mean(loss_rhythm)   
                reduced_bar_loss = tf.reduce_mean(loss_bar)            
                tf.summary.scalar('loss', reduced_loss)
                tf.summary.scalar('note_loss', reduced_note_loss)
                tf.summary.scalar('rhythm_loss', reduced_rhythm_loss)
                tf.summary.scalar('bar_loss', reduced_bar_loss)
                if l2_regularization_strength is None:
                    return self.y, pd, reduced_loss
                else:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if not('bias' in v.name)])
                    total_loss = reduced_loss +l2_regularization_strength * l2_loss
                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)
                    return self.y, pd, total_loss
if __name__ == "__main__":
    ####define graph####
    net = SampleRnnModel_w_mode_switch(batch_size=5, big_frame_size=8, \
                                       frame_size = 2 , rnn_type= "GRU", \
                                       dim = 256, n_rnn = 1, seq_len = 32,\
                                       mode_choice = "bar_note", if_cond = True, \
                                       piano_dim = 200, note_channel = 100, rhythm_channel =50, chord_channel = 50, alpha = 0.5)   
    print(net.big_frame_cell)