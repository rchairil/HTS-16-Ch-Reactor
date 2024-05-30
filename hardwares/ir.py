# ******************************************************************************
# hardware                : ir
# language                : python
# requirement             : --
# author                  : Lu Wang, Chemical Engineering, University of Southern California
# function                : used in main script for reading IR signals; this file is not needed for three-reagent, single-phase production. 
# ******************************************************************************

import time
from pyfirmata import ArduinoMega as Arduino 
from pyfirmata import util
import multiplexer as MUX
import pandas as pd

import serial
from serial.tools.list_ports import comports
 
for comport in comports():
        if comport.serial_number == '95530343235351605281':
                board_port = comport.device
 
# Create the Serial port for ArduinoMega board
board = Arduino(board_port) 
# Wait for 2 seconds for the communication to get established
time.sleep(2) 
 
# first multiplexer, define switch and signal pins on boards
PinS, PinSig = MUX.DefinePin(board, [30,31,32,33],3)
# second multiplexer, define switch and signal pins on boards
PinS2, PinSig2 = MUX.DefinePin(board, [40,41,42,43],4)
 
# need this iteration otherwise pin just reporting 'None'
it = util.Iterator(board) 
it.start()
time.sleep(1)
 
channels = 32 # 32 channels in total
MUX.SwitchMUX(PinS, 0) # start switches by connecting to channel 0
MUX.SwitchMUX(PinS2, 0)
Data = {} # Create an empty dictionary
num_of_data = 2000 # collect 2000 data points each channel
 
 
for n in range(channels):
        print("Reading channel "+str(n+1)+"...")
        t = []
        v = []
        t0 = time.time()
        if n<16:        # It's on the first mutiplexer
                MUX.SwitchMUX(PinS,n)
                time.sleep(0.1)
                for m in range(num_of_data):
                        t.append(time.time()-t0)
                        v.append(MUX.Read(PinSig))
                        time.sleep(0.000001) # record every 1 microsecond
                Data['Time'+str(n+1)] = t
                Data['Chan'+str(n+1)] = v
               
# switch to the second mutiplexer, need to re-define channel No.
        else: 
                m = n-16
                MUX.SwitchMUX(PinS2,m)
                time.sleep(0.1)
                for m in range(num_of_data):
                        t.append(time.time()-t0)
                        v.append(MUX.Read(PinSig2))
                        time.sleep(0.000001) # record every 1 microsecond
                Data['Time'+str(n+1)] = t
                Data['Chan'+str(n+1)] = v

df = pd.DataFrame.from_dict(Data, orient="index")
df.to_csv("ir.csv")