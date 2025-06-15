#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.sparse import spdiags
from scipy import signal
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees,degrees2kilometers
import time
from scipy import interpolate

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append('/home/yaolinm/Projects/Mexico2/')
from funcs.utils import *

def obspy_filter(data,lowcut,highcut,fs,order=4,axis=-1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    z, p, k = signal.iirfilter(order, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = signal.zpk2sos(z, p, k)
    new_data=signal.sosfilt(sos, data,axis=axis)
    return new_data

def lowpass_square_filter(data,freqmax,fs):
    
    freqs=np.fft.fftfreq(len(tr),1/100)
    filt=np.zeros(len(freqs))
    filt[np.where((freqs>=0) & (freqs<=freqmax))]=1
    
    return np.fft.ifft(np.fft.fft(data)*filt)

def shift_time(cc2d,sampling_rate=20):
    mid=(cc2d.shape[-1]-1)/2
    index=np.argmax(cc2d,axis=-1)
    return (index-mid)/sampling_rate

def shift_tr(tr,time,sampling_rate=25):
    
    omega=np.fft.fftfreq(len(tr),1/sampling_rate)*2*np.pi
    
    return np.fft.ifft(np.fft.fft(tr)*np.exp(-1j*omega*time)).real

#mccc

def mccc(synthesis_df):
    n_chs=len(synthesis_df)
    n_pairs=int(n_chs*(n_chs-1)/2)+1
    sigmat=np.zeros(n_pairs)

    start=0
    for i in range(n_chs-1):
        for j in range(i+1,n_chs):        
            correlations=np.correlate(synthesis_df[i,:],synthesis_df[j,:],mode='full')
            sigmat[start]=(synthesis_df.shape[1]-1-np.argmax(correlations))
            start+=1

    A=np.zeros((n_pairs,n_chs))
    A[-1,:]=1
    init_pos,init_neg=0,1

    for i in range(n_pairs-1):

        A[i,init_pos]=1
        A[i,init_neg]=-1

        if init_neg<n_chs-1:
            init_neg+=1

        else:
            init_pos+=1
            init_neg=init_pos+1
        
    return A, sigmat

def corrnow(invmatrix, invdata, half_len):
    """ 
    z without weighting:
            t = inv(A'A) * A' * dt = 1/n * A' * dt, where A'A = nI
    """
    nsta = np.shape(invmatrix)[1]
    invmodel = np.dot(np.transpose(invmatrix), invdata)
    invmodel /= nsta
    
    for i in range(len(invmodel)):
        if invmodel[i]<=-half_len:
            invmodel[i]+=2*half_len
        elif invmodel[i]>=half_len:
            invmodel[i]-=2*half_len
        
    return invmodel

def iterative_mccc(df,start,end,iterations=10):
    
    it=0
    solutions=np.zeros((iterations,len(df)))
    to_mccc=df.copy()
    
    while it<iterations:
        iteration=normalize_df(to_mccc)[:,start:end]
        A,sigmat=mccc(iteration)
        solution=corrnow(A,-sigmat,int((end-start)/2))
        for i in range(len(to_mccc)):
            to_mccc[i,:]=np.roll(to_mccc[i,:],int(-solution[i]))
        solutions[it,:]=solution
        it+=1
        
    return to_mccc,solutions