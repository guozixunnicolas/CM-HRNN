from collections import defaultdict
import copy
from math import log, floor, ceil
import pprint
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from midiutil import MIDIFile
import numpy as np
import random
import os
import glob

DEBUG = False
PITCHES = xrange(21,109,1)
PITCHES_MAP = { p : i for i, p in enumerate(PITCHES) } 
'''{21: 0, 22: 1, 23: 2, 24: 3, 25: 4, 26: 5, 27: 6, 28: 7, 29: 8, 30: 9, 31: 10, 32: 11, 33: 12, 34: 13, 35: 14, 36: 15, 37: 16, 38: 17, 39: 18, 40: 19, 41: 20, 42: 21, 43: 22, 44: 23, 45: 24, 46: 25, 47: 26, 48: 27, 49: 28, 50: 29, 51: 30, 52: 31, 53: 32, 54: 33, 55: 34, 56: 35, 57: 36, 58: 37, 59: 38, 60: 39, 61: 40, 62: 41, 63: 42, 64: 43, 65: 44, 66: 45, 67: 46, 68: 47, 69: 48, 70: 49, 71: 50, 72: 51, 73: 52, 74: 53, 75: 54, 76: 55, 77: 56, 78: 57, 79: 58, 80: 59, 81: 60, 82: 61, 83: 62, 84: 63, 85: 64, 86: 65, 87: 66, 88: 67, 89: 68, 90: 69, 91: 70, 92: 71, 93: 72, 94: 73, 95: 74, 96: 75, 97: 76, 98: 77, 99: 78, 100: 79, 101: 80, 102: 81, 103: 82, 104: 83, 105: 84, 106: 85, 107: 86, 108: 87}'''


PITCHES_EVENT = xrange(20,109,1)
PITCHES_MAP_EVENT = { p : i for i, p in enumerate(PITCHES_EVENT) }
MAX_NOTE_LEN = 64
'''{20: 0, 21: 1, 22: 2, 23: 3}'''
OUT_FOLDER_pianoroll = "stop_sustain_mid"
OUT_FOLDER_event = "event_based"
def nearest_pow2(x):
    '''Normalize input to nearest power of 2, or midpoints between
    consecutive powers of two. Round down when halfway between two
    possibilities.'''

    low = 2**int(floor(log(x, 2)))
    high = 2**int(ceil(log(x, 2)))
    mid = (low + high) / 2

    if x < mid:
        high = mid
    else:
        low = mid
    if high - x < x - low:
        nearest = high
    else:
        nearest = low
    return nearest
    
def quantize_tick(tick, ticks_per_quarter, quantization):
    '''Quantize the timestamp or tick.

    Arguments:
    tick -- An integer timestamp
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as 1/2**quantization
    '''
    assert (ticks_per_quarter * 4) % 2 ** quantization == 0, \
        'Quantization too fine. Ticks per quantum must be an integer.'
    ticks_per_quantum = (ticks_per_quarter * 4) / float(2 ** quantization)
    quantized_ticks = int(
        round(tick / float(ticks_per_quantum)) * ticks_per_quantum)
    return quantized_ticks

def validate_data(path, quant):
    '''Creates a folder containing valid MIDI files.
    Arguments:
    path -- Original directory containing untouched midis.
    quant -- Level of quantisation'''

    path_prefix, path_suffix = os.path.split(path) 
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0
    mid_lst = []
    cond_lst = []
    paired_lst = []
    for root, dirs, files in os.walk(path): #root: ./data, dirs:[], files: all mid files
        for file in files:
            if file.split('.')[-1] == 'mid' or file.split('.')[-1] == 'MID':
                total_file_count += 1
                #print ('Processing ' + str(file))
                midi_path = os.path.join(root,file)
                try:
                    midi_file = MidiFile(midi_path)
                    _ , _=get_note_track(midi_file)
                except (KeyError, IOError, TypeError, IndexError, EOFError, ValueError):
                    #print ("Bad MIDI.")
                    continue
                time_sig_msgs = [ msg for msg in midi_file.tracks[0] if msg.type == 'time_signature' ]
                #<meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>
                if len(time_sig_msgs) == 1:
                    time_sig = time_sig_msgs[0]
                    if not (time_sig.numerator == 4 and time_sig.denominator == 4):
                        #print ('\tTime signature not 4/4. Skipping ...')
                        continue
                else:
                    #print ('\tNo time signature. Skipping ...')
                    continue

                mid = quantize(MidiFile(os.path.join(root,file)), quant)
                if not mid:
                    #print ('Invalid MIDI. Skipping...')
                    continue
                out_file = os.path.join(root, file)
 
                if out_file.split("_")[-1]=="condition.mid":
                    cond_lst.append(out_file)
                else:
                    mid_lst.append(out_file) 

                processed_count += 1
    
    #print ('\nProcessed {} files out of {}'.format(processed_count, total_file_count))
    for f in mid_lst:
        midi_num = f.split("/")[-1].split(".")[0] #midi1
        candidate = os.path.join( os.path.join(*f.split("/")[0:-1]), 
                         midi_num+"_condition.mid")
        if candidate in cond_lst:
            paired_lst.append((f, candidate))
            #print("validated pair:" , f, candidate)
    return paired_lst
def quantize_data(path, quant):
    '''Creates a folder containing the quantised MIDI files.

    Arguments:
    path -- Validated directory containing midis.
    quant -- Level of quantisation'''

    path_prefix, path_suffix = os.path.split(path)

    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0

    base_path_out = os.path.join(path_prefix, path_suffix+'_quantized')
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1] == 'mid' or file.split('.')[-1] == 'MID':
                total_file_count += 1
                mid = quantize(MidiFile(os.path.join(root,file)),quant)
                if not mid:
                    print( 'Invalid MIDI. Skipping...')
                    continue
                suffix = root.split(path)[-1]
                out_dir = base_path_out + '/' + suffix
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(out_dir, file)

                #print ('Saving', out_file)
                mid.save(out_file)

                processed_count += 1

    #print( 'Processed {} files out of {}'.format(processed_count, total_file_count))

def save_data(paired_lst, quant, choice = "pianoroll"):
    total_file_count = 0
    processed_count = 0
    sustain_statics = defaultdict(list)
    if len(paired_lst)>0:
        genre,_ = paired_lst[0]
        genre = genre.split("/")[1]
        sustain_statics["genre"] = genre
    for midi_file, cond_file in paired_lst:
        total_file_count += 2
        try:
            if choice =="pianoroll":
                midi_array = midi_to_stop_sustain(midi_file, quant)
                cond_array = midi_to_stop_sustain(cond_file, quant)
                midi_array, cond_array = pad2eqlen(midi_array, cond_array)
            elif choice == "event":
                midi_array,sustain_statics,if_valid = midi_to_event(midi_file,quant,sustain_statics)
                midi_array = midi_array[1:]
                #print(midi_file,midi_array.shape)
                if if_valid is False:
                    print("skip",midi_file)
                    continue
        except:
            #print("except ", midi_file, cond_file )
            continue

        genre = midi_file.split("/")[1]
        number = midi_file.split("/")[-1].split(".")[0][4:]

        midi_name = genre + number + ".npy" #vocal1.npy
        cond_name = genre + number + "_condition.npy" #vocal1_condtion.npy
        

        if choice =="pianoroll":
            midi_dir = os.path.join(OUT_FOLDER_pianoroll, genre, "audio" )
            cond_dir = os.path.join(OUT_FOLDER_pianoroll, genre, "condition" )
            if not os.path.exists(midi_dir):
                os.makedirs(midi_dir)
            if not os.path.exists(cond_dir):
                os.makedirs(cond_dir)
            np.save(os.path.join(midi_dir, midi_name), midi_array)
            np.save(os.path.join(cond_dir, cond_name), cond_array)
            print ('files updated', os.path.join(midi_dir, midi_name), os.path.join(cond_dir, cond_name))
            processed_count += 2

        elif choice == "event":
            midi_dir = os.path.join(OUT_FOLDER_event, genre)
            #cond_dir = os.path.join(OUT_FOLDER_event, genre, "condition" )
            if not os.path.exists(midi_dir):
                os.makedirs(midi_dir)
            #if not os.path.exists(cond_dir):
                #os.makedirs(cond_dir)
            np.save(os.path.join(midi_dir, midi_name), midi_array) 
            #print ('files updated', os.path.join(midi_dir, midi_name))
            processed_count += 1       

    if choice == "event":

        process_statics(sustain_statics)


        
    #print ('\nProcessed {} files out of {}'.format(processed_count, total_file_count))

def process_statics(sustain_statics, if_save = True):
    genre = sustain_statics["genre"]
    note_len = []
    times_occured = [] 
    total_times = 0
    for length in sustain_statics:
        if length != "genre":
            total_times+=len(sustain_statics[length])
    #print(total_times)
    for length in sustain_statics:
        if length != "genre":
            note_len.append(str(length))
            prob = float(len(sustain_statics[length]))/total_times
            #print(length,"happens",len(sustain_statics[length]), "times in", genre)
            times_occured.append(prob)
    plt.xticks(fontsize=5)
    plt.bar(note_len, times_occured)
    plt.suptitle(genre)
    plt.xlabel("all notes' sustain length")
    plt.ylabel("prob")
    if if_save:
        img_path=os.path.join(OUT_FOLDER_event, genre)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_name = os.path.join(img_path, "{}.png".format(genre))
        plt.savefig(img_name, bbox_inches="tight", dpi=gcf().dpi)
        print("status updated at", img_name)
    plt.clf()
def add_zero(sustain_midi, stop_array):
    #sustain_midi :(len, 89)
    #stop array :(len, 1)
    zero_timing = []
    for row, t in enumerate(sustain_midi):
        if not np.any(t): #this row contains all zero 
            zero_timing.append(row)

    if len(zero_timing) == 0:
        stop_sustain_midi = np.concatenate((stop_array , sustain_midi), axis= -1)
    else:
        for idx in zero_timing:
            stop_array[idx, :] = 1
        stop_sustain_midi = np.concatenate((stop_array , sustain_midi), axis= -1)
    return stop_sustain_midi

def midi_to_event(midi_file, quantization, sustain_statics, trim_beginning = True):

    mid = MidiFile(midi_file)
    time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
    assert len(time_sig_msgs) == 1, 'No time signature found'
    time_sig = time_sig_msgs[0]
    assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'
    mid = quantize(mid, quantization=quantization)

    _, track = get_note_track(mid)
    ticks_per_quarter = mid.ticks_per_beat
    time_msgs = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_msgs])
    track_len_ticks = cum_times[-1]
    notes = [
        (time * (2**quantization/4) / (ticks_per_quarter),  ########maybe it means which bar /4means 16th note?
        msg.type, 
        msg.note, 
        msg.velocity)
        for (time, msg) in zip(cum_times, time_msgs)
        if msg.type == 'note_on' or msg.type == 'note_off']

    num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))

    normalized_num_steps = num_steps
    notes.sort(key=lambda(position, note_type, note_num, velocity):(position,velocity))
 
    open_msgs = defaultdict(list)
    midi_array = np.zeros((1, len(PITCHES_EVENT)+ MAX_NOTE_LEN))
    if_valid = True
    prev_note_off_pos = 0
    if trim_beginning:
        position, note_type, note_num, velocity = notes[0]
        if position!= 0:
            prev_note_off_pos = position


    for (position, note_type, note_num, velocity) in notes:
        
        note_matrix = np.zeros((1, len(PITCHES_EVENT)))
        sustain_matrix = np.zeros((1, MAX_NOTE_LEN))

        if note_type == "note_on" and velocity > 0:
            #if prev note off time != current note on time, append rest
            if position!= prev_note_off_pos:
                sustain_len = position - prev_note_off_pos
                current_note = PITCHES_MAP_EVENT[20]
                sustain_statics[sustain_len].append(str(0)) ##update statics
                if sustain_len>MAX_NOTE_LEN:
                    print("ignore this file, exceeding max sustain len",midi_file,position, current_note, sustain_len)
                    if_valid = False
                    break
                ##write to matrix##
                note_matrix[:,current_note] = 1
                sustain_matrix[:, sustain_len]  =1
                final_slice = np.concatenate((note_matrix , sustain_matrix), axis= -1)
                midi_array = np.concatenate((midi_array , final_slice), axis= 0)

            open_msgs[note_num].append((position, note_type, note_num, velocity))
            

        elif note_type == 'note_off' or (note_type == 'note_on' and velocity == 0):
            current_note = PITCHES_MAP_EVENT[note_num]
            note_on_open_msgs = open_msgs[note_num]

            if len(note_on_open_msgs) == 0:
                return
            stack_pos, _, _, _ = note_on_open_msgs[0] 

            open_msgs[note_num] = note_on_open_msgs[1:] #FIFO QUEUE, GET RID OF THE NOTE ON IN EARLIER POSOTION

            sustain_len = position - stack_pos
            if sustain_len>MAX_NOTE_LEN:
                print("ignore this file, exceeding max sustain len",midi_file,position, current_note, sustain_len)
                if_valid = False
                break
            ##write to matrix##
            note_matrix[:,current_note] = 1
            sustain_matrix[:, sustain_len]  =1
            final_slice = np.concatenate((note_matrix , sustain_matrix), axis= -1)
            midi_array = np.concatenate((midi_array , final_slice), axis= 0)
            ##update statics##
            sustain_statics[sustain_len].append(str(0))
            prev_note_off_pos = position
    
    return midi_array , sustain_statics, if_valid


def midi_to_stop_sustain(midi_file, quantization):
    mid = MidiFile(midi_file)

    time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
    assert len(time_sig_msgs) == 1, 'No time signature found'
    time_sig = time_sig_msgs[0]
    assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'

    # Quantize the notes to a grid of time steps.
    mid = quantize(mid, quantization=quantization)

    # Convert the note timing and velocity to an array.
    _, track = get_note_track(mid)
    ticks_per_quarter = mid.ticks_per_beat
    time_msgs = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_msgs])
    track_len_ticks = cum_times[-1]
    if DEBUG:
        print ('Track len in ticks:', track_len_ticks)
    notes = [
        (time * (2**quantization/4) / (ticks_per_quarter),  ########maybe it means which bar /4means 16th note?
        msg.type, 
        msg.note, 
        msg.velocity)
        for (time, msg) in zip(cum_times, time_msgs)
        if msg.type == 'note_on' or msg.type == 'note_off']
    '''
    (2, 'note_on', 60, 100)
    (6, 'note_on', 60, 100)
    (6, 'note_on', 60, 0)
    (10, 'note_on', 60, 100)
    (10, 'note_on', 60, 0)
    (12, 'note_on', 60, 100)
    (12, 'note_on', 60, 0)'''

    num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))

    normalized_num_steps = num_steps
    
    notes.sort(key=lambda(position, note_type, note_num, velocity):(position,-velocity))
 
    midi_array = np.zeros((normalized_num_steps, len(PITCHES)))
    sustain_array = np.zeros((normalized_num_steps, 1))
    stop_array = np.zeros((normalized_num_steps, 1))
    open_msgs = defaultdict(list)

    for (position, note_type, note_num, velocity) in notes:
        #print("processing", position, note_type, note_num, velocity)
        if position == normalized_num_steps:
            #print ('Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1))
            position = normalized_num_steps - 1
            # continue

        if position > normalized_num_steps:
            # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
            continue

        if note_type == "note_on" and velocity > 0:
            open_msgs[note_num].append((position, note_type, note_num, velocity))
            midi_array[position, PITCHES_MAP[note_num]] = 1

        elif note_type == 'note_off' or (note_type == 'note_on' and velocity == 0):

            note_on_open_msgs = open_msgs[note_num]
            
            if len(note_on_open_msgs) == 0:
                #print ('Bad MIDI, Note has no end time.')
                return
            stack_pos, _, _, vel = note_on_open_msgs[0] 
            open_msgs[note_num] = note_on_open_msgs[1:] #FIFO QUEUE, GET RID OF THE NOTE ON IN EARLIER POSOTION
            current_pos = position - 1
            while current_pos > stack_pos:  #sustain note from last note on to current note off
                sustain_array[current_pos, :] = 1
                current_pos -= 1
    sustain_midi  = np.concatenate((sustain_array , midi_array), axis= -1) 
    stop_sustain_midi = add_zero(sustain_midi, stop_array)

    """ for (position, note_type, note_num, velocity) in notes:
        if position == normalized_num_steps:
            print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
            position = normalized_num_steps - 1
            # continue

        if position > normalized_num_steps:
            # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
            continue
        if note_type == "note_on" and velocity > 0:
            open_msgs[note_num].append((position, note_type, note_num, velocity))
            midi_array[position, PITCHES_MAP[note_num]] = 1"""

    return stop_sustain_midi

def quantize(mid, quantization=5):
    '''Return a midi object whose notes are quantized to
    1/2**quantization notes.

    Arguments:
    mid -- MIDI object
    quantization -- The note duration, represented as
      1/2**quantization.'''

    quantized_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    note_track_idx, note_track = get_note_track(mid)
    new_track = quantize_track( note_track, mid.ticks_per_beat, quantization) #ticks per beat(quarter note) 220
    if new_track == None:
        return None
    quantized_mid.tracks[note_track_idx] = new_track
    return quantized_mid

def get_note_track(mid):
    '''Given a MIDI object, return the first track with note events.'''

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on':
                return i, track
    #raise ValueError(
        #'MIDI object does not contain any tracks with note messages.')

def quantize_track(track, ticks_per_quarter, quantization): #ticks per beat 220, quantization 5
    '''Return the differential time stamps of the note_on, note_off, and
    end_of_track events, in order of appearance, with the note_on events
    quantized to the grid given by the quantization.

    Arguments:
    track -- MIDI track containing note event and other messages
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as
      1/2**quantization.'''

    pp = pprint.PrettyPrinter()

    # Message timestamps are represented as differences between
    # consecutive events. Annotate messages with cumulative timestamps.

    # Assume the following structure:
    # [header meta messages] [note messages] [end_of_track message]
    first_note_msg_idx = None
    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    cum_msgs = zip(
        np.cumsum([msg.time for msg in track[first_note_msg_idx:]]), #The time attribute of each message is the number of seconds since the last message or the start of the file.
                  [msg for msg in track[first_note_msg_idx:]])
    
    end_of_track_cum_time = cum_msgs[-1][0]
    '''(3410, <message note_on channel=0 note=64 velocity=100 time=3410>)
    (3630, <message note_on channel=0 note=64 velocity=0 time=220>)
    (3630, <message note_on channel=0 note=67 velocity=100 time=0>)
    (3850, <message note_on channel=0 note=67 velocity=0 time=220>)
    '''
    quantized_track = MidiTrack()
    quantized_track.extend(track[:first_note_msg_idx])
    # Keep track of note_on events that have not had an off event yet.
    # note number -> message
    open_msgs = defaultdict(list)
    quantized_msgs = []
    for cum_time, msg in cum_msgs:
        if DEBUG:
            #print('Message:', msg)
            #print( 'Open messages:')
            pp.pprint(open_msgs)
        if msg.type == 'note_on' and msg.velocity > 0:
            # Store until note off event. Note that there can be
            # several note events for the same note. Subsequent
            # note_off events will be associated with these note_on
            # events in FIFO fashion.
            open_msgs[msg.note].append((cum_time, msg))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # assert msg.note in open_msgs, \
            #     'Bad MIDI. Cannot have note off event before note on event'

            if msg.note not in open_msgs:
                 #print('Bad MIDI. Cannot have note off event before note on event')
                 return

            note_on_open_msgs = open_msgs[msg.note]

            if len(note_on_open_msgs) == 0:
                #print('Bad MIDI, Note has no end time.')
                return

            # assert len(note_on_open_msgs) > 0, 'Bad MIDI, Note has no end time.'

            note_on_cum_time, note_on_msg = note_on_open_msgs[0]
            open_msgs[msg.note] = note_on_open_msgs[1:]

            # Quantized note_on time
            quantized_note_on_cum_time = quantize_tick(
                note_on_cum_time, ticks_per_quarter, quantization)

            # The cumulative time of note_off is the quantized
            # cumulative time of note_on plus the orginal difference
            # of the unquantized cumulative times.
            quantized_note_off_cum_time = quantized_note_on_cum_time + (cum_time - note_on_cum_time)
            quantized_msgs.append((min(end_of_track_cum_time, quantized_note_on_cum_time), note_on_msg))
            quantized_msgs.append((min(end_of_track_cum_time, quantized_note_off_cum_time), msg))

            if DEBUG:
                print('Appended', quantized_msgs[-2:])
        elif msg.type == 'end_of_track':
            quantized_msgs.append((cum_time, msg))

        if DEBUG:
            print ('\n')

    # Now, sort the quantized messages by (cumulative time,
    # note_type), making sure that note_on events come before note_off
    # events when two event have the same cumulative time. Compute
    # differential times and construct the quantized track messages.
    quantized_msgs.sort(
        key=lambda (cum_time, msg): cum_time
        if (msg.type=='note_on' and msg.velocity > 0) else cum_time + 0.5)

    diff_times = [quantized_msgs[0][0]] + list(
        np.diff([ msg[0] for msg in quantized_msgs ]))
    for diff_time, (cum_time, msg) in zip(diff_times, quantized_msgs):
        quantized_track.append(msg.copy(time=diff_time))
    if DEBUG:
        #print ('Quantized messages:')
        pp.pprint(quantized_msgs)
        pp.pprint(diff_times)
    return quantized_track

def pad2eqlen(midi_array, cond_array):
    if midi_array.shape!= cond_array.shape:
        print("shape conflict: mid:{}, cond:{}, now padding".format(midi_array.shape, cond_array.shape))
    if midi_array.shape[0]>cond_array.shape[0]:
        pad_len = midi_array.shape[0] - cond_array.shape[0]
        pad = np.zeros((pad_len, cond_array.shape[1]))
        pad[:, 0 ] = 1
        cond_array = np.concatenate((cond_array, pad), axis = 0)
        print("after padding, mid {}, cond {}".format(midi_array.shape, cond_array.shape))

    elif midi_array.shape[0]<cond_array.shape[0]:
        pad_len = cond_array.shape[0] - midi_array.shape[0]
        pad = np.zeros((pad_len, midi_array.shape[1])) 
        pad[:, 0 ] = 1
        midi_array = np.concatenate((midi_array, pad),axis = 0)
        print("after padding, mid {}, cond {}".format(midi_array.shape, cond_array.shape))
    return midi_array, cond_array



if __name__ == '__main__':
    new_path = "PROCESSED_TEST_MIDI"
    #subdir = glob.glob('RAW_MIDI/**/')
    subdir = glob.glob("RAW_MIDI/**/")
    
    for dirs in subdir:
        paired_lst = validate_data(dirs, quant=4)
        if len(paired_lst)!=0:
            save_data(paired_lst, quant=4, choice = "event")
        else:
            print("no files in",dirs)
    #print(len(PITCHES_EVENT)+ MAX_NOTE_LEN)



        
