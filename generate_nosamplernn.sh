#!/bin/bash
python generation_w_mode_switch.py \
	--gen_dir=./test/generated_result \
    --cond_dir=data_inC_bar_pad/test/Electronic\
	--logdir_root=logdir/11_05_2020_08_29_00_Electronic_nosamplernn/model.ckpt-120000\
	--rhythm_temp=0.6\
	--note_temp=0.6