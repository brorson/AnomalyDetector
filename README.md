This is an implementation of an anomaly detector based on
subspace projection methods.  It looks for changes in sounds detected
by a microphone.  You first train it on the "normal" sound of e.g. a
motor, then you let it run a loop while it listens for deviations in
the received sound from the training sound. 

The code runs on a BeagleBone, with an ADC-001 A/D cape, 
which is a 2-channel A/D board designed for data acquisition in the
audio range (50Hz -- 15kHz).  The cape takes input from a MIC-001
microphone. Complete documentation for the ADC-001 cape and MIC-001
microphone is available under: 

https://github.com/brorson/ADC-001_hardware_information

These hardware items are available for sale at BNM-Hobbies:

http://www.bnm-hobbies.com/store/index.php?main_page=index&cPath=150

Stuart Brorson
March, 2018.
