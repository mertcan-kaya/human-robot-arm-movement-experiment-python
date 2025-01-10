#!/usr/bin/env python
# Copyright (c) 2020, Universal Robots A/S,
# Modified by Mertcan Kaya (2022-02-14)

# example_control_loop_debug

import sys
sys.path.append('..')
import logging

import os
from datetime import datetime
import csv
import trajectory as trj
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

##def format_time():
##    t = datetime.now()
##    s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
##    return s[:-3]

#logging.basicConfig(level=logging.INFO)

# configure recipes (not used here)
ROBOT_HOST = '10.8.0.231'#'localhost' 
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'

# recording options
record_external = False
record_internal = True

## Change for each participant
participant_num = 1 # staring from 1, increase by 1
mot_num = 0 # 0,1,2,3

print('Experiment #' + str(participant_num))
#print('remove-mot_num: ' + str(mot_num))

## Enum

# motion direction
HOR = 0 # horizontal
VER = 1 # vertical

## Via-Points in time

# Robot;Human
A = np.array([[VER],[VER]])
B = np.array([[VER],[HOR]])
C = np.array([[HOR],[VER]])
D = np.array([[HOR],[HOR]])

# sequences
mot_seq = np.zeros((2,4,4))
mot_seq[:,:,0] = np.concatenate((D,C,A,B), axis=1)
mot_seq[:,:,1] = np.concatenate((A,D,B,C), axis=1)
mot_seq[:,:,2] = np.concatenate((B,A,C,D), axis=1)
mot_seq[:,:,3] = np.concatenate((C,B,D,A), axis=1)

ms_num = (participant_num-1) % 4
if mot_seq[0,mot_num,ms_num] == VER and mot_seq[1,mot_num,ms_num] == VER:
    mot_str = 'A'
    print('Motion sequence ' + mot_str)
    print(' Robot: Vertical')
    print(' Human: Vertical')
elif mot_seq[0,mot_num,ms_num] == VER and mot_seq[1,mot_num,ms_num] == HOR:
    mot_str = 'B'
    print('Motion sequence ' + mot_str)
    print(' Robot: Vertical')
    print(' Human: Horizontal')
elif mot_seq[0,mot_num,ms_num] == HOR and mot_seq[1,mot_num,ms_num] == VER:
    mot_str = 'C'
    print('Motion sequence ' + mot_str)
    print(' Robot: Horizontal')
    print(' Human: Vertical')
elif mot_seq[0,mot_num,ms_num] == HOR and mot_seq[1,mot_num,ms_num] == HOR:
    mot_str = 'D'
    print('Motion sequence ' + mot_str)
    print(' Robot: Horizontal')
    print(' Human: Horizontal')
else:
    mot_str = 'ERROR!'

speed_ratio = np.array([0.5,1,1.5])

vel_seq = np.zeros((3,6))
vel_seq[:,0] = np.array([2,1,0])
vel_seq[:,1] = np.array([1,2,0])
vel_seq[:,2] = np.array([2,0,1])
vel_seq[:,3] = np.array([0,1,2])
vel_seq[:,4] = np.array([1,0,2])
vel_seq[:,5] = np.array([0,2,1])

speed_num = vel_seq.shape[0]
prm_num = vel_seq.shape[1]

##trn_time    = 5 # transition time
##wait_time   = 3 # waiting time
##btwn_time   = 1 # between time
##ft_time     = 2 # one full-swing time

trn_time    = 3 # transition time
wait_time   = 0 # waiting time
btwn_time   = 0 # between time
ft_time     = 2 # one full-swing time

ft_dist = np.deg2rad(35) # one full-swing distance

cycle_num = 3

## Trajectory interpolation

tstp = 0.008

## Declaration

sequence_num = 1

point_num = sequence_num*(prm_num*speed_num*(2*cycle_num+1)+2)+2

Pos = np.zeros((6,point_num))
Vel = np.zeros((6,2))
Acc = np.zeros((6,2))

Ctrn = np.zeros((6,6,sequence_num+1))
Cswg = np.zeros((6*2*cycle_num,6,prm_num*sequence_num*speed_num))
   
T = np.zeros((1,point_num))     
Ttrn = np.array([0,trn_time])
Tswg = np.zeros((1,2*cycle_num+1,prm_num*sequence_num*speed_num))
Tstep = np.zeros((1,sequence_num*(2*prm_num*speed_num+2)+2))
TOstep = np.zeros((1,sequence_num*4+2))

## Point generation

deg1 = 23.5928 # asind(0.08535/0.21325)

# initial point
count = 0
T[:,count] = 0
step_count = 0
Tstep[:,step_count] = T[:,count]
ostep_count = 0
TOstep[:,ostep_count] = T[:,count]
Pos[:,count] = np.deg2rad([-90,-180,deg1,180-deg1,-90,0])

# for p in range(sequence_num):
p = 0

# transition
count = count + 1
T[:,count] = T[:,count-1] + trn_time
step_count = step_count + 1
Tstep[:,step_count] = T[:,count]
ostep_count = ostep_count + 1
TOstep[:,ostep_count] = T[:,count]
Pos[:,count] = Pos[:,0]
if mot_seq[0,mot_num,ms_num] == HOR:
    Pos[1,count] = Pos[1,0] - ft_dist # -: left, +: right
else:
    Pos[0,count] = Pos[0,0] + ft_dist # -: top, +: bottom
Ptrn = Pos[:,count-1:count+1]
for j in range(6):
    Ctrn[:,j,p] = np.squeeze(trj.minimumJerkCoefficient1DOF(Ttrn,Ptrn[j,:],Vel[j,:],Acc[j,:]))

# wait
count = count + 1
T[:,count] = T[:,count-1] + wait_time
step_count = step_count + 1
Tstep[:,step_count] = T[:,count]
ostep_count = ostep_count + 1
TOstep[:,ostep_count] = T[:,count]
Pos[:,count] = Pos[:,count-1]

# permutation
for r in range(prm_num):
    for w in range(speed_num):
        # cycle
        for i in range(2*cycle_num):
            # half-motion
            tp_time = ft_time*speed_ratio[int(vel_seq[w,r])]
            
            count = count + 1
            T[:,count] = T[:,count-1] + tp_time
            Pos[:,count] = Pos[:,0]
            if i % 2 == 0:
                if mot_seq[0,mot_num,ms_num] == HOR:
                    Pos[1,count] = Pos[1,0] + ft_dist # +: to right
                else:
                    Pos[0,count] = Pos[0,0] - ft_dist # +: to bottom
            if i % 2 == 1:
                if mot_seq[0,mot_num,ms_num] == HOR:
                    Pos[1,count] = Pos[1,0] - ft_dist # -: to left
                else:
                    Pos[0,count] = Pos[0,0] + ft_dist # -: to top
        step_count = step_count + 1
        Tstep[:,step_count] = T[:,count]
        if w == speed_num-1 and r == prm_num-1:
            ostep_count = ostep_count + 1
            TOstep[:,ostep_count] = T[:,count]
        Tswg[:,:,(p*prm_num+r)*speed_num+w] = T[:,count-2*cycle_num:count+1]-T[:,count-2*cycle_num]
        Pswg = Pos[:,count-2*cycle_num:count+1]
        for j in range(6):
            Cswg[:,j,(p*prm_num+r)*speed_num+w] = np.squeeze(trj.minimumJerkCoefficient1DOF(np.squeeze(Tswg[:,:,(p*prm_num+r)*speed_num+w]),Pswg[j,:],Vel[j,:],Acc[j,:]))
        
        # wait
        count = count + 1
        if w == speed_num-1 and r == prm_num-1:
            T[:,count] = T[:,count-1] + wait_time
            ostep_count = ostep_count + 1
            TOstep[:,ostep_count] = T[:,count]
        else:
            T[:,count] = T[:,count-1] + btwn_time
        step_count = step_count + 1
        Tstep[:,step_count] = T[:,count]
        Pos[:,count] = Pos[:,count-1]

# last transition
count = count + 1
T[:,count] = T[:,count-1] + trn_time
step_count = step_count + 1
Tstep[:,step_count] = T[:,count]
ostep_count = ostep_count + 1
TOstep[:,ostep_count] = T[:,count]
Pos[:,count] = Pos[:,0]
Ptrn = Pos[:,count-1:count+1]
for j in range(6):
    Ctrn[:,j,sequence_num] = np.squeeze(trj.minimumJerkCoefficient1DOF(Ttrn,Ptrn[j,:],Vel[j,:],Acc[j,:]))

total_step = int(T[:,point_num-1]/tstp+1)
t = np.squeeze(T[:,0])

t_step = t
stepcount_out = 1
stepcount_in = 1
trncount = 0
swgcount = 0
cyccount = 0
r = 0

swing_state = 0

if record_external == True or record_internal == True:
    # detect the current working directory and print it
    patha = os.getcwd()
    print ("The current working directory is %s" % patha)
    
    # define the name of the directory to be created
    save_path = "participant " + str(participant_num) + "-nortde"
    pathb = patha + '/' + save_path

    try:
        os.mkdir(pathb)
    except OSError:
        print ("Creation of the directory %s failed" % pathb)
    else:
        print ("Successfully created the directory %s " % pathb)
if record_external == True:
    # run record.py
    extProc = sp.Popen(['python','record.py'])
if record_internal == True:
    #l_time = time.localtime()
    #current_date = time.strftime('%Y-%m-%d', l_time)
    now = datetime.now()
    d_string = now.strftime("%Y-%m-%d %H.%M.%S")
    filename = str(mot_num+1) + '-' + mot_str + ' ' + d_string + '.csv'
    #completeName = os.path.join(save_path, filename)
    completeName = save_path + '\\' + filename
    # log actual joint pos
    outfile = open(completeName, 'w', newline='')
    writer = csv.writer(outfile, delimiter=',')
    list_time = ['time']
    list_state = ['state']
    #list_target_q = ['target_q0', 'target_q1', 'target_q2', 'target_q3', 'target_q4', 'target_q5']
    #list_target_qd = ['target_qd0', 'target_qd1', 'target_qd2', 'target_qd3', 'target_qd4', 'target_qd5']
    list_actual_q = ['actual_q0', 'actual_q1', 'actual_q2', 'actual_q3', 'actual_q4', 'actual_q5']
    writer.writerow(list_time+list_state+list_actual_q)

q_pos = Pos[:,0]

keep_running = True
output_int_register_0 = 0

idx = 1
# control loop
while keep_running:

    runtime_state = 2
    
    
    # do something...
    if runtime_state == 2:
        
        if output_int_register_0 == 0:
            setlist = [q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]]
            output_int_register_0 = 1
        else:
            if idx < total_step:                
                if record_internal == True:
                    now = datetime.now()
                    tm_string = now.strftime("%S.%f")
                    # write data
                    writer.writerow([tm_string, stepcount_in, q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]])
##                print(idx)
                
                if t > Tstep[:,stepcount_in]:
                    
                    if t > TOstep[:,stepcount_out]:
                        stepcount_out = stepcount_out + 1
                        swing_state = 0
                        if stepcount_out % 4 == 1:
                            trncount = trncount + 1
                            cyccount = 0
                        elif stepcount_out % 4 == 3:
                            swing_state = 1
                        t_step = 0

                    stepcount_in = stepcount_in + 1
                    if swing_state == 1:
                        cyccount = cyccount + 1
                        if cyccount % 2 == 1:
                            swgcount = swgcount + 1
                            t_step = 0

                if stepcount_out % 4 == 1:
                    # transition
                    for j in range(6):
                        q_pos[j] = trj.minimumJerkPolynomial1DOF(t_step,Ttrn,np.squeeze(Ctrn[:,j,trncount]))
                elif stepcount_out % 4 == 3:
                    if cyccount % 2 == 1:
                        # swing
                        for j in range(6):
                            q_pos[j] = trj.minimumJerkPolynomial1DOF(t_step,np.squeeze(Tswg[:,:,swgcount-1]),np.squeeze(Cswg[:,j,swgcount-1]))
                    else:
                        # wait
                        for j in range(6):
                            q_pos[j] = q_pos[j]
                else:
                    # wait
                    for j in range(6):
                        q_pos[j] = q_pos[j]
                        
                t_step = round(t_step + tstp,3)
                t = round(t + tstp,3)

                setlist = [q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]]
                idx = idx + 1
            else:
                setlist = [q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]]
                
                # trajectory command
                keep_running = False
           
if record_internal == True:
    now = datetime.now()
    tm_string = now.strftime("%S.%f")
    # write data
    writer.writerow([tm_string, stepcount_in, q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]])
    
    # shut down
    outfile.close()

print('Disconnected')
