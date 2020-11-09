#!/bin/bash
python generation_w_mode_switch.py \
	--gen_dir=./test/generated_result \
    --cond_dir=data_inC_bar_pad/test/Electronic\
	--logdir_root=logdir/11_08_2020_16_25_26_Electronic_note/model.ckpt-120000\
	--rhythm_temp=0.1\
	--note_temp=0.7