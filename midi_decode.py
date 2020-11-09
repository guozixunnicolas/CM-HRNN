from collections import defaultdict
import copy
from math import log, floor, ceil
import pprint
import matplotlib.pyplot as plt
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import random
import os
import glob

ticks_per_quarter = 220
quant=4

def time_to_tick(np_time):
    tick = np_time*ticks_per_quarter/(2**quant/4)
    return tick


def decode(np_file):
    print("decoding {}".format(np_file))
    filenp = np.load(np_file) #(len. dim)
    
    length , piano_dim= filenp.shape
    zero_start = np.zeros([1,piano_dim])
    zero_end = np.zeros([1,piano_dim])
    filenp= np.concatenate((zero_start,filenp))
    filenp= np.concatenate((filenp, zero_end))


    diff = np.diff(filenp,axis = 0)
    #diff_trans = np.transpose(diff)
    velocity_changes = np.nonzero(diff) 

    FIFO_note_on = defaultdict(list)
    new_mid = MidiFile(ticks_per_beat=ticks_per_quarter) 
    track = MidiTrack()
    new_mid.tracks.append(track)
    for time, note in zip(*velocity_changes):
        on_o_off = diff[time,note]
        print(on_o_off)
        note += 21
        if on_o_off == 1:
            FIFO_note_on[note].append(time)

        elif on_o_off==-1:
            current_time = time
            previous_time = FIFO_note_on[note][0]
            FIFO_note_on[note] = FIFO_note_on[note][1:]
            ##encode midi message##    


            on_time = time_to_tick(previous_time)
            off_time = time_to_tick(current_time)
            #print("ontime{},tick{},offtime{},tick{}".format(previous_time,on_time,current_time,off_time))
            #print(note, on_time, off_time)
            track.append(Message('note_on', note=note, velocity=100, time=0))
            track.append(Message('note_off', note=note, velocity=100, time=off_time-on_time))
            #print("writing note:{} from {} to {}".format(note, on_time, off_time)) 
        name = os.path.split(np_file)[-1].split('.')[0]
        new_mid.save('{}_try.mid'.format(name))


if __name__ == '__main__':
    
    np_files = glob.glob("*.npy")
    
    for np_file in np_files:
        decode(np_file)










