#!/usr/bin/env python
# Copyright (c) 2020, Universal Robots A/S,
# Modified by Mertcan Kaya (2022-02-14)

import sys
sys.path.append('..')
import logging

import os
from datetime import datetime
import csv
import trajectory as trj
import numpy as np
import subprocess as sp

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

#logging.basicConfig(level=logging.INFO)

# configure recipes
ROBOT_HOST = '10.8.0.231'#'localhost' 
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'

# recording options
record_external = False
record_internal = True

## Change when appearance changes
humanoid = 0 # 0: nonhumanoid, 1: humanoid

## Change for each participant
participant_num = 40 # staring from 1, increase by 1
mot_num = 2 # 0,1,2,3

print('Experiment #' + str(participant_num))

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

if humanoid == 1:
    print('Humainod')
else:
    print('Nonhumanoid')

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

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')
setp_names, setp_types = conf.get_recipe('setp')
command_names, command_types = conf.get_recipe('command')
watchdog_names, watchdog_types = conf.get_recipe('watchdog')

# connect to controller
con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

# get controller version
con.get_controller_version()

print('Connected to robot')

# setup recipes
con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
command = con.send_input_setup(command_names, command_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

if record_external == True or record_internal == True:
    # detect the current working directory and print it
    patha = os.getcwd()
    print ("The current working directory is %s" % patha)

    if humanoid == 1:
        hmnd_chr = "H"
    else:
        hmnd_chr = "N"
    
    # define the name of the directory to be created
    save_path = "participant " + str(participant_num) + hmnd_chr
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
    now = datetime.now()
    d_string = now.strftime("%Y-%m-%d %H.%M.%S")
    filename = str(mot_num+1) + '-' + mot_str + ' ' + d_string + '.csv'
    completeName = save_path + '\\' + filename
    # log actual joint pos
    outfile = open(completeName, 'w', newline='')
    writer = csv.writer(outfile, delimiter=',')
    list_time = ['time']
    list_state = ['state']
    list_actual_q = ['actual_q0', 'actual_q1', 'actual_q2', 'actual_q3', 'actual_q4', 'actual_q5']
    writer.writerow(list_time+list_state+list_actual_q)

q_pos = Pos[:,0]

# initialize joint pos 
setp.input_double_register_0 = q_pos[0]
setp.input_double_register_1 = q_pos[1]
setp.input_double_register_2 = q_pos[2]
setp.input_double_register_3 = q_pos[3]
setp.input_double_register_4 = q_pos[4]
setp.input_double_register_5 = q_pos[5]

# the function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0

# functions
def setp_to_list(setp):
    list = []
    for i in range(0,6):
        list.append(setp.__dict__["input_double_register_%i" % i])
    return list

def list_to_setp(setp, list):
    for i in range(0,6):
        setp.__dict__["input_double_register_%i" % i] = list[i]
    return setp

def print_list(plist):
    for i in range(0,len(plist)):
        if i < len(plist)-1:
            print("%7.4f" % plist[i], end=", ")
        else:
            print("%7.4f" % plist[i], end="\n")
        
# start data synchronization
if not con.send_start():
    sys.exit()

# joint_space
command.input_int_register_1 = 0

# init_joint
command.input_int_register_2 = 0
        
# trajectory command
command.input_int_register_3 = 1

con.send(command)

print('Press play to start')

keep_running = True

idx = 1
# control loop
while keep_running:
    # receive the current state
    state = con.receive()
    
    if state is None:
        break
    
    # do something...
    if state.runtime_state == 2:

        if state.output_int_register_0 == 0:
            setlist = [q_pos[0], q_pos[1], q_pos[2], q_pos[3], q_pos[4], q_pos[5]]
        else:
            if idx < total_step:                
                if record_internal == True:
                    now = datetime.now()
                    tm_string = now.strftime("%H.%M.%S.%f")
                    # write data
                    writer.writerow([tm_string, stepcount_in, state.actual_q[0], state.actual_q[1], state.actual_q[2], state.actual_q[3], state.actual_q[4], state.actual_q[5]])

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
                command.input_int_register_3 = 0
                con.send(command)

                keep_running = False
            
        list_to_setp(setp, setlist)
        # send new setpoint        
        con.send(setp)
        
    # kick watchdog
    con.send(watchdog)

if record_internal == True:
    now = datetime.now()
    tm_string = now.strftime("%H.%M.%S.%f")
    # write data
    writer.writerow([tm_string, stepcount_in, state.actual_q[0], state.actual_q[1], state.actual_q[2], state.actual_q[3], state.actual_q[4], state.actual_q[5]])
    
    # shut down
    outfile.close()

print('Disconnected')

con.send_pause()
con.disconnect()

if record_external == True:
    # terminate record.py
    sp.Popen.terminate(extProc)
