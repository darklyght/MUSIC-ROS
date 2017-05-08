#!/usr/bin/env python

import os
import glob
import rospy
from music.msg import Directions
import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
message = Directions()
def music():
    global message
    
    # Get last created file
    path = 'C:/Users/darklyght/Desktop/'
    if any(File.endswith('.csv') for File in os.listdir(path)):
        newest = max(glob.iglob(path + '*.csv'), key=os.path.getctime)
    time = np.genfromtxt(newest, delimiter = ',', max_rows = 1)
    
    ##### EDIT STUFF HERE TO TEST #####
    SampleSize = 100 # Size of data sample
    SamplingRate = 100000.0/time # Sampling rate (include decimal place)
    BandpassLow = 29000 # Lower limit of Butterworth bandpass filter
    BandpassHigh = 31000 # Upper limit of Butterworth bandpass filter
    BandpassOrder = 5 # Butterworth filter order
    
    
    
    ##### Data sampling #####
    start = 1
    x = np.transpose(np.genfromtxt(newest, delimiter = ',', skip_header = start, max_rows = SampleSize))
    while (np.std(x[0], ddof=1) < 20):
        start += 100
        x = np.transpose(np.genfromtxt(newest, delimiter = ',', skip_header = start, max_rows = SampleSize))
    
    if (x.size != 0 and len(x[0]) >= SampleSize):
        x[0] = (x[0] - np.mean(x[0])) / np.std(x[0])
        x[1] = (x[1] - np.mean(x[1])) / np.std(x[1])
        x[2] = (x[2] - np.mean(x[2])) / np.std(x[2])
        
        # Sample characteristics
        L = SampleSize # Sample size
        rate = SamplingRate # Sampling rate
        step = np.arange(0, L / rate, 1 / rate) # Time step
        freq = 30000 # Frequency of wave 
        speed = 343.0 # Speed of wave
        M = 1 # Number of sources
        N = 3 # Number of sensors
        
        # Array geometry
        r = np.array([[0., 0., 0.], [0., 0.064, 0.], [0.072, 0., 0.]])



        ##### Signal Pre-processing #####
        # Bandpass filter
        nyq = 0.5 * (SamplingRate)
        low = BandpassLow / nyq
        high = BandpassHigh / nyq
        b,a = sig.butter(BandpassOrder, [low, high], 'bandpass')
        x[0] = sig.lfilter(b,a,x[0])
        x[1] = sig.lfilter(b,a,x[1])
        x[2] = sig.lfilter(b,a,x[2])
                
        # Theoretical phase calculations
        #k = np.zeros(shape=(3, 1))
        #az = 273
        #el = 116
        #k[0, 0] = (2 * np.pi / (speed / freq)) * np.cos(az * np.pi / 180) * np.sin(el * np.pi / 180)
        #k[1, 0] = (2 * np.pi / (speed / freq)) * np.sin(az * np.pi / 180) * np.sin(el * np.pi / 180)
        #k[2, 0] = (2 * np.pi / (speed / freq)) * np.cos(el * np.pi / 180)
        #A = np.exp(-1j * np.matmul(r, k))
        #print(np.angle(A))

        # Fourier transformation
        FFT = fft.fft(x)
        freqs = np.fft.fftfreq(x[0].size, d=1/SamplingRate)
        index = np.where(freqs >= 30000)[0][0]
        
        # Signal with phase information at desired frequency
        W = np.matrix([[FFT[0,:][index]], [FFT[1,:][index]], [FFT[2,:][index]]])

        
        
        ##### Music Algorithm #####
        # Sample covariance matrix
        Rxx = W * np.matrix.getH(W) / L

        # Eigendecompose
        [D, E] = np.linalg.eigh(Rxx)
        D = np.real(D)
        idx = D.argsort()[::1]
        lmbd = D[idx]
        E = E[:, idx]
        En = E[:, 0:len(E)-M]
        En = np.array(En)
        
        # MUSIC search directions
        ElRange, AzRange = np.meshgrid(np.arange(90, 181, 1), np.arange(0, 361, 1))
        Z = np.zeros(shape=AzRange.shape)
        kSearch = np.array([[(2 * np.pi / (speed / freq)) * np.cos(AzRange * np.pi / 180) * np.sin(ElRange * np.pi / 180)], [(2 * np.pi / (speed / freq)) * np.sin(AzRange * np.pi / 180) * np.sin(ElRange * np.pi / 180)], [(2 * np.pi / (speed / freq)) * np.cos(ElRange * np.pi / 180)]])
        kSearch = np.swapaxes(kSearch, 0, 2)
        kSearch = np.swapaxes(kSearch, 1, 3)
        ASearch = np.exp(-1j * np.matmul(r, kSearch))
        Z = np.transpose(np.reshape(np.sum(np.square(np.absolute(np.matmul(np.matrix.conj(np.swapaxes(ASearch, 2, 3)), En))), axis=3), (361,91)))
        P, Q = np.unravel_index(Z.argmin(), Z.shape)

       
       
        ##### Display #####
        # Plot
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(AzRange, ElRange, -np.log10(np.transpose(Z)))
        #plt.show()
        #plt.pcolor(Z)
        #plt.show()
        message.azimuth = AzRange[P]
        message.elevation = ElRange[Q]
        
def publisher():
    global message

    pub = rospy.Publisher('pinger', Directions, queue_size=1)
    rospy.init_node('MUSIC', anonymous=False)
    pub_rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        music()
        rospy.loginfo(message)
        pub.publish(message)
        pub_rate.sleep()        

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
