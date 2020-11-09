import os
import glob
import subprocess
import math
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
    out = subprocess.run(["java", "-jar","omnisia.jar","-i",mid_file,"-draw"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = out.stdout.decode('utf-8')
    log_folder = txt.split("\n")[5].split(":")[-1][1:]
    cos_file = glob.glob(log_folder+"/*.cos")[0]
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
            #print("long pattern exists in ", log_folder)
            current_file_has_long_pattern = True
        else:
            current_file_has_long_pattern = False
            subprocess.run(["rm", "-rf",log_folder])
        num_short_pattern = sum([x[1] for x in span_occurence if x[0]<long_short_threshold])
        

    return ratio, num_long_pattern, num_short_pattern, current_file_has_long_pattern





if __name__ == '__main__':
    folders_2b_evalutated = ["./test/generated_result/10_26_2020_11_14_22_Electronic_note/120000/8/0.7_0.8", "test/generated_result/10_24_2020_13_39_33_Electronic_nosamplernn/120000/16/0.6_0.6"]
    lst_of_long_pattern = []
    for folder in folders_2b_evalutated:
        mid_files_lst = glob.glob(folder+"/*.mid")
        mid_files_lst2 = [f for f in mid_files_lst if f.split("_")[-1]=="eva.mid"]

        #evaluate a folder of files
        feature_4_each_file = [get_feature(mid_file, long_short_threshold= 64) for mid_file in mid_files_lst2]
        ratio_4_folder = sum([x[0]for x in feature_4_each_file])/len(mid_files_lst2)
        long_patterns_4_folder = sum([x[1]for x in feature_4_each_file])
        short_patterns_4_folder = sum([x[2]for x in feature_4_each_file])
        mask_lst = [x[3] for x in feature_4_each_file]
        files_with_long_pattern = [x for x, y in zip(mid_files_lst2, mask_lst) if y is True]
        print(folder,"ratio:",ratio_4_folder, "long pattern:",long_patterns_4_folder, "short pattern:",short_patterns_4_folder, "files_with_long_pattern", files_with_long_pattern)
        

