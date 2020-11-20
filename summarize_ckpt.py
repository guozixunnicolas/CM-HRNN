import pandas as pd
import glob
import os
from collections import defaultdict
import json
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import subprocess
import math
from samplernn import decode_rhythm
def get_empty_dict():
    dictionary= {
        "note_tmp":[],
        "rhythm_tmp":[],
        "comp_ratio":[],
        "sucessbar_ratio":[],
        "long_pattern":[],
        "short_pattern":[],
        "logdir":[],
        "ckpt":[],
        "mode_choice":[],
        "big_frame_size":[],
        "frame_size":[],
        "if_cond":[],
        "rnn_type":[],
        "rnn_num":[],
        "rnn_dim":[],
        "alpha1":[],
        "alpha2":[]

    }
    return dictionary

def check_sucess_bar_ratio(npy_files, rhythm_channel, chord_channel):
    fail_count = 0
    sucess_count = 0
    for f in npy_files:
        events = np.load(f)[0]
        bar_info = events[:,:2]
        event_info = events[:, 2:]
        bar_event = np.zeros((1,2))
        bar_event[:,1] = 1

        bar_event_idx_lst = list(np.where((bar_info == bar_event).all(axis = 1))[0])
        prev_total_len = 0

        while len(bar_event_idx_lst)!=0 and len(bar_event_idx_lst)>1:
            start = bar_event_idx_lst[0]
            end = bar_event_idx_lst[1]
            for idx in range(start, end):
                rhythm_info_at_idx = np.argmax(event_info[idx][chord_channel:chord_channel+rhythm_channel])

                event_rhythm_decoded = decode_rhythm(rhythm_info_at_idx)
                prev_total_len += float(event_rhythm_decoded)

            if prev_total_len%4!=0:
                fail_count +=1
            else:
                sucess_count+=1
            prev_total_len = 0
            bar_event_idx_lst = bar_event_idx_lst[1:]
    sucess_ratio = sucess_count/(fail_count+sucess_count)
    return sucess_ratio

def parenthetic_contents(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            if len(stack)==1:
                yield (len(stack), string[start + 1: i])

def get_feature(mid_file, long_short_threshold = 16):
    try:
        out = subprocess.run(["java", "-jar","omnisia.jar","-i",mid_file,"-draw"],stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout= 30)
    except :
        print("item is probably empty return 1, 0 , 0 False")
        return 1, 0, 0, False
    txt = out.stdout.decode('utf-8')
    log_folder = txt.split("\n")[5].split(":")[-1][1:]
    cos_file = glob.glob(log_folder+"/*.cos")[0]
    if not os.path.isfile(cos_file):
        print("cos file doesn't exist")
        return 1, 0, 0, False
    """png_file = glob.glob(log_folder+"/*.png")[0]
    if not os.path.isfile(png_file):
        print("png file doesn't exist")
        return 1, 0, 0, False"""
    ratio = None
    file_lst_with_long_pattern = []
    with open(cos_file, "r") as f:
        lines = f.readlines()
        ratio = [float(line.split(" ")[-1]) for line in lines if line.startswith("compressionRatio ")][0]
        if math.isnan(ratio):
            ratio = 1
        pattern_occurence = [(list(parenthetic_contents(line))[0][1], list(parenthetic_contents(line))[1][1].count("v")) for line in lines if line.startswith("T")]
        #span: number of 16th notes 220:16th note, 880: 1/4note, 3520: 1note
        span_occurence = [(  (float(p_o[0].split("p")[-1].split(",")[0][1:])- float(p_o[0].split("p")[1].split(",")[0][1:]))/220
                              ,p_o[1]
                          ) for p_o in pattern_occurence if p_o[1]!=1]
        num_long_pattern= sum([x[1] for x in span_occurence if x[0]>=long_short_threshold])
        if num_long_pattern>0:
            mid_file_with_chord = mid_file[:-8]+".mid"
            current_file_has_long_pattern = True
        else:
            current_file_has_long_pattern = False
            subprocess.run(["rm", "-rf",log_folder])
        num_short_pattern = sum([x[1] for x in span_occurence if x[0]<long_short_threshold])
    
    return ratio, num_long_pattern, num_short_pattern, current_file_has_long_pattern

folders_2b_evaluated = ["test/generated_result/11_14_2020_13_23_50_Electronic_ad_rm3t/120000/32/0.6_0.4",
                       "test/generated_result/11_14_2020_13_23_50_Electronic_ad_rm3t/120000/32/0.6_0.2",
                       "test/generated_result/11_14_2020_13_23_50_Electronic_ad_rm3t/120000/32/0.6_0.1",
                       "test/generated_result/11_14_2020_13_22_48_Electronic_ad_rm3t/120000/16/0.6_0.4",
                       "test/generated_result/11_14_2020_13_22_48_Electronic_ad_rm3t/120000/16/0.6_0.2",
                       "test/generated_result/11_14_2020_13_22_48_Electronic_ad_rm3t/120000/16/0.6_0.1",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.7_0.1",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.7_0.2",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.7_0.4",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.6_0.1",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.6_0.2",
                       "test/generated_result/11_14_2020_13_19_10_Electronic_ad_rm3t/120000/32/0.6_0.4",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.7_0.1",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.7_0.2",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.7_0.4",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.6_0.1",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.6_0.2",
                       "test/generated_result/11_14_2020_13_17_51_Electronic_ad_rm3t/120000/16/0.6_0.4",
                       "test/generated_result/11_10_2020_08_15_26_Electronic_ad_rm2t/120000/16/0.7_0.1",
                       "test/generated_result/11_10_2020_08_15_26_Electronic_ad_rm2t/120000/16/0.7_0.3",
                       "test/generated_result/11_10_2020_08_15_26_Electronic_ad_rm2t/120000/16/0.7_0.4",
                       "test/generated_result/11_17_2020_07_19_37_Electronic_ad_rm2t_birnn/120000/16/0.7_0.4",
                       "test/generated_result/11_17_2020_07_19_37_Electronic_ad_rm2t_birnn/120000/16/0.7_0.2",
                       "test/generated_result/11_17_2020_07_19_37_Electronic_ad_rm2t_birnn/120000/16/0.7_0.1",]
merged_dict = []
for folder in folders_2b_evaluated:

    #dictionary = get_empty_dict()
    dictionary = {}
    folder_name = folder.replace("/","_") #"test_generated_result_11_14_2020_13_23_50_Electronic_ad_rm3t_120000_32_0.6_0.1"
    json_name = "test/"+"_".join(folder_name.split("_")[3:])+".json"
    if os.path.isfile(json_name):
        with open(json_name, "r") as js:
            dictionary = json.load(js)
    else:
        npy_files_lst = glob.glob(folder+"/*.npy")
        mid_files_lst = glob.glob(folder+"/*.mid")
        mid_files_eva = [f for f in mid_files_lst if f.split("_")[-1]=="eva.mid"]

        logdir = folder.split("/")[-4] #10_24_2020_13_39_33_Electronic_nosamplernn
        ckpt = folder.split("/")[-3] #12000
        note_tmp = folder.split("/")[-1].split("_")[0] #0.6
        rhythm_tmp = folder.split("/")[-1].split("_")[1] #0.6
        dictionary["logdir"]= logdir
        dictionary["ckpt"]=ckpt
        dictionary['note_tmp']=note_tmp
        dictionary["rhythm_tmp"]=rhythm_tmp

        config_file_for_folder = "logdir/"+logdir+"/config.txt"
        with open(config_file_for_folder,"r") as f_op:
            lines = f_op.readlines()
            big_frame_size = lines[1].split(":")[-1][1:-1]
            frame_size = lines[2].split(":")[-1][1:-1]
            mode_choice_ckpt = lines[4].split(":")[-1][1:-1]
            if_cond_ckpt = lines[5].split(":")[-1][1:-1]
            no_rnn = lines[6].split(":")[-1][1:-1]
            note_channel = lines[7].split(":")[-1][1:-1]
            rhythm_channel = lines[8].split(":")[-1][1:-1]
            chord_channel = lines[12].split(":")[-1][1:-1]
            alpha1 = lines[13].split(":")[-1][1:-1]
            alpha2 = lines[17].split(":")[-1][1:-1]
            rnn_type = lines[9].split(":")[-1][1:-1]
            dim = lines[10].split(":")[-1][1:-1]
        dictionary["rnn_dim"]=dim
        dictionary["mode_choice"]=mode_choice_ckpt
        if mode_choice_ckpt =="bar_note" or mode_choice_ckpt =="ad_rm3t":    
            dictionary["big_frame_size"]=big_frame_size
            dictionary["frame_size"]=frame_size
        else:
            dictionary["big_frame_size"]="NA"
            dictionary["frame_size"]=frame_size
        dictionary["if_cond"]=if_cond_ckpt  
        dictionary["rnn_type"]=rnn_type
        dictionary["rnn_num"]=no_rnn
        dictionary["alpha1"]=alpha1
        dictionary["alpha2"]=alpha2

        #check sucess bar ratio
        sucessbar_ratio_4_folder = check_sucess_bar_ratio(npy_files_lst, int(rhythm_channel), int(chord_channel))
        dictionary["sucessbar_ratio"]=sucessbar_ratio_4_folder

        #check comp ratio, long short pattern
        feature_4_each_file = [get_feature(mid_file, long_short_threshold= 64) for mid_file in mid_files_eva]
        
        ratio_4_folder = sum([x[0]for x in feature_4_each_file])/len(mid_files_eva)
        long_patterns_4_folder = sum([x[1]for x in feature_4_each_file])
        short_patterns_4_folder = sum([x[2]for x in feature_4_each_file])

        dictionary["comp_ratio"]=ratio_4_folder
        dictionary["long_pattern"]=long_patterns_4_folder
        dictionary["short_pattern"]=short_patterns_4_folder



        json_out = json.dumps(dictionary)
        f = open(json_name,"w")
        f.write(json_out)
        f.close()
    merged_dict.append(dictionary)
#print(merged_dict)
df = pd.DataFrame.from_dict(merged_dict)
"""merged_data_frame = {}
for k in dictionary.keys():
  merged_data_frame[k]=tuple(d[k] for d in merged_dict)
"""
#df = pd.DataFrame(merged_dict, columns = ["logdir","comp_ratio","long_pattern","short_pattern", "mode_choice", "big_frame_size", "frame_size"])

print(df[ df["mode_choice"]=="ad_rm2t_birnn"])

#print(df[ 'mode_choice','long_pattern'])



