# Hierarchical Recurrent Neural Networks for Conditional Melody Generation with Long-term Structure



## Description

A conditional melody generative system. Paper link: 
## Getting Started

### Results and data format
Kindly check the result demo folder for the generated results. For input data format check the npy files in the result demo folder. 

### Dependencies

* tensorflow 1.14
* midiutil
* numpy
* pandas

### Getting started

* Training script: run_xx.sh; Generation script: generate_xx.sh; Evaluation script: comp_successbar_ratio.py
* 2t_fc: 2 tier without acc. time info;
* 3t_fc: 3 tier without acc. time info;
* adrm2t_fc: 2 tier with acc. time info;
* adrm3t_fc: 3 tier with acc. time info;
* adrm3t_fc_rs: 3 tier with acc. time info with residual conn.;
* bln_attn_fc: baseline attentionrnn;
* bln_fc: baseline vanila rnn;

