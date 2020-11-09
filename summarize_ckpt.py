import pandas as pd
import glob
import os
from collections import defaultdict
files = glob.glob("logdir/**/*.txt")
test_result = glob.glob("test/*.txt")
#columns: folder, mode_choice, big_size, size, data_used, 
#if_cond, rnn_type, rnn_num, rnn_dim, alpha, chord_channel, note_channel, rhythm_channel
dictionary= {
    "folder":[],
    "ckpt":[],
    "note_tmp":[],
    "rhythm_tmp":[],
    "ratio":[],
    "long_pattern":[],
    "short_pattern":[],
    "mode_choice":[],
    "big_frame_size":[],
    "frame_size":[],
    "if_cond":[],
    "rnn_type":[],
    "rnn_num":[],
    "rnn_dim":[],
    "alpha":[]
}
"""for f in files:
    folder_name = f.split("/")[1]
    dictionary["folder"].append(folder_name)
    with open(f,"r") as f_op:
        lines = f_op.readlines()
        big_frame_size = lines[1].split(":")[-1][1:-1]
        frame_size = lines[2].split(":")[-1][1:-1]
        mode_choice_ckpt = lines[4].split(":")[-1][1:-1]
        if_cond_ckpt = lines[5].split(":")[-1][1:-1]
        no_rnn = lines[6].split(":")[-1][1:-1]
        note_channel = lines[7].split(":")[-1][1:-1]
        rhythm_channel = lines[8].split(":")[-1][1:-1]
        chord_channel = lines[12].split(":")[-1][1:-1]
        alpha = lines[13].split(":")[-1][1:-1]
        rnn_type = lines[9].split(":")[-1][1:-1]
        dim = lines[10].split(":")[-1][1:-1]
    dictionary["rnn_dim"].append(dim)
    dictionary["mode_choice"].append(mode_choice_ckpt)
    if mode_choice_ckpt =="bar_note":    
        dictionary["big_frame_size"].append(big_frame_size)
        dictionary["frame_size"].append(frame_size)
    else:
        dictionary["big_frame_size"].append("NA")
        dictionary["frame_size"].append(frame_size)
    dictionary["if_cond"].append(if_cond_ckpt)  
    dictionary["rnn_type"].append(rnn_type)
    dictionary["rnn_num"].append(no_rnn)
    dictionary["alpha"].append(alpha)   """
    

for g in test_result:
    with open(g,"r") as g_op:    
        lines = g_op.readlines()    
        logdir = lines[0].split(":")[-1][:-1]
        ckpt = lines[1].split(":")[-1][:-1]
        note_tmp = lines[2].split(":")[-1][:-1]
        rhythm_tmp = lines[3].split(":")[-1][:-1]
        ratio = lines[4].split(":")[-1][:-15]
        long_pattern = lines[5].split(":")[-1][:-1]
        short_pattern = lines[6].split(":")[-1][:-1]
    dictionary["folder"].append(logdir)
    dictionary["ckpt"].append(ckpt)
    dictionary['note_tmp'].append(note_tmp)
    dictionary["rhythm_tmp"].append(rhythm_tmp)
    dictionary["ratio"].append(ratio)
    dictionary["long_pattern"].append(long_pattern)
    dictionary["short_pattern"].append(short_pattern)

    logdir_config = "logdir/{}/config.txt".format(logdir)
    #print(logdir, ckpt, note_tmp, rhythm_tmp, ratio, long_pattern, short_pattern, os.path.isfile(logdir_config))
    with open(logdir_config,"r") as f_op:
        lines = f_op.readlines()
        big_frame_size = lines[1].split(":")[-1][1:-1]
        frame_size = lines[2].split(":")[-1][1:-1]
        mode_choice_ckpt = lines[4].split(":")[-1][1:-1]
        if_cond_ckpt = lines[5].split(":")[-1][1:-1]
        no_rnn = lines[6].split(":")[-1][1:-1]
        note_channel = lines[7].split(":")[-1][1:-1]
        rhythm_channel = lines[8].split(":")[-1][1:-1]
        chord_channel = lines[12].split(":")[-1][1:-1]
        alpha = lines[13].split(":")[-1][1:-1]
        rnn_type = lines[9].split(":")[-1][1:-1]
        dim = lines[10].split(":")[-1][1:-1]
    dictionary["rnn_dim"].append(dim)
    dictionary["mode_choice"].append(mode_choice_ckpt)
    if mode_choice_ckpt =="bar_note":    
        dictionary["big_frame_size"].append(big_frame_size)
        dictionary["frame_size"].append(frame_size)
    else:
        dictionary["big_frame_size"].append("NA")
        dictionary["frame_size"].append(frame_size)
    dictionary["if_cond"].append(if_cond_ckpt)  
    dictionary["rnn_type"].append(rnn_type)
    dictionary["rnn_num"].append(no_rnn)
    dictionary["alpha"].append(alpha)
#df = pd.DataFrame(dictionary, columns = ["folder","ckpt","note_tmp","rhythm_tmp","ratio","long_pattern","short_pattern", "mode_choice","if_cond", "big_frame_size", "frame_size", "rnn_num", "rnn_dim","rnn_type", "alpha"])
df = pd.DataFrame(dictionary, columns = ["folder","ratio","long_pattern","short_pattern", "mode_choice","if_cond", "big_frame_size", "frame_size", "rnn_num", "rnn_dim", "alpha"])

print(df[ df["mode_choice"]=="nosamplernn"])


