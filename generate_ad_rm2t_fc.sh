#!/bin/bash
python generation_w_mode_switch_v2.py \
	--gen_dir=./test/generated_result \
    --cond_dir=data_inC_bar_pad_mged_split/test/merged\
	--logdir_root=logdir/01_12_2021_03_27_45_merged_ad_rm2t_fc/model.ckpt-120000\
	--rhythm_temp=0.4\
	--note_temp=0.7