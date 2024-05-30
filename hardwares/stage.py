import serial
import time
from serial.tools.list_ports import comports
from pyfirmata import ArduinoMega
 
for comport in comports():
	SN = comport.serial_number
	if SN == '55739323637351517151':
		board_port = comport.device 
 
# Create the Serial port for ArduinoMega board
board = ArduinoMega(board_port) 
# Wait for 2 seconds for the communication to get established
time.sleep(2) 
 
PinDir = board.get_pin('d:7:o') # output digital pin
PinStep = board.get_pin('d:8:o') # output digital pin
 
pulseWidthSec = 1/1000000000.0 # unit second, this is 1 microsecond
SecBetweenSteps = 2/1000000000.0 # larger results in slower steps
block_steps = 4060 # number of steps the motor needs to travel one block
 
 
# start by not moving the stage
PinStep.write(0)
 
# ask the user the moving direction
direct = input('Which direction to move? \n0. Away from motor\n1. Towards motor\n')
 
# start moving the stage for one block
numOfSteps = block_steps*1 # one block

if int(direct) == 0:
	print('away from motor for '+str(block_steps)+' blocks')
	PinDir.write(0) # away from the motor
	for n in range (numOfSteps):
		PinStep.write(1)
		time.sleep(pulseWidthSec)
		PinStep.write(0)
		time.sleep(SecBetweenSteps)
elif int(direct) == 1:        
	print('approach to the motor for '+str(block_steps)+' blocks')        
	PinDir.write(1) # approaching to the motor
	for n in range (numOfSteps):
		PinStep.write(1)
		time.sleep(pulseWidthSec)
		PinStep.write(0)
		time.sleep(SecBetweenSteps)
 
board.exit()