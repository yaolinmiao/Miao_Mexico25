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

def cal_sensor_location(azimuth,distance,source_location=(0,0)):
    
    """
    
    Calculate the location of the virtual receiver, based on geometric parameters
    
    azimuth: degree from the source to the sensor
    distance: linear distnace
    source_location: two-item tuple, (x-direction,y-direction)
    
    Return: sensor location in tuple: (x-direction,y-direction)
    
    """
    azimuth_rad = math.radians(azimuth)
    delta_x = distance * math.sin(azimuth_rad)
    delta_y = distance * math.cos(azimuth_rad)
    
    x = source_location[0] + delta_x
    y = source_location[1] + delta_y
    
    return (x,y)

def calculate_point(reference_point, azimuth, distance):

    azimuth_rad = math.radians(azimuth)

    delta_x = distance * math.sin(azimuth_rad)
    delta_y = distance * math.cos(azimuth_rad)

    x_new = reference_point[0] + delta_x
    y_new = reference_point[1] + delta_y

    return x_new, y_new

def ricker_taup(T,sampling,f0,amp,shift=0):
    
    t=np.arange(0,T,1/sampling)
    tau=np.pi*f0*(t-1.5/f0)
    q=amp*(1.0-2.0*tau**2.0)*np.exp(-tau**2)
    
    return np.roll(q,int(shift*sampling)),t

def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def normalize(data):
    return data/np.max(data)

def normalize_df(df):
    
    new_df=np.zeros_like(df)
    for i in range(len(df)):
        new_df[i,:]=normalize(df[i,:])
    
    return new_df

def calculate_azimuth(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    azimuth = math.atan2(x, y)

    azimuth_degrees = (math.degrees(azimuth) + 360) % 360
    return azimuth_degrees-180

def roll_time(origin,sensor,stf,sampling=20,speed=1):
    
    t=((sensor[0]-origin[0])**2+(sensor[1]-origin[1])**2)**0.5/speed
    
    rolled=np.roll(stf,int(t*sampling))
    
    return rolled

def shift_tr(tr,time,sampling_rate=25):
    
    omega=np.fft.fftfreq(len(tr),1/sampling_rate)*2*np.pi
    
    return np.fft.ifft(np.fft.fft(tr)*np.exp(-1j*omega*time)).real

def shift_tr_vectorized(tr, time, sampling_rate=50):
    """
    Vectorized version of shift_tr that applies phase shifts to all traces at once.
    
    Parameters:
    - tr: (n_stations, window_size) → Input time-domain signals.
    - time: (n_stations, x_mesh, y_mesh) → Time shifts for each station and grid point.
    - sampling_rate: Sampling rate in Hz.

    Returns:
    - Shifted traces with shape (window_size, n_stations, x_mesh, y_mesh)
    """
    n_stations, window_size = tr.shape
    x_mesh_size, y_mesh_size = time.shape[1:]  # Extract grid dimensions

    omega = np.fft.fftfreq(window_size, 1 / sampling_rate) * 2 * np.pi  
    tr_fft = np.fft.fft(tr, axis=1).T  # Transpose to (window_size, n_stations)
    tr_fft = tr_fft[:, :, None, None]  # Now (window_size, n_stations, 1, 1)
    phase_shift = np.exp(-1j * omega[:, None, None, None] * time)
    shifted_tr = np.fft.ifft(tr_fft * phase_shift, axis=0).real  # (window_size, n_stations, x_mesh, y_mesh)

    return shifted_tr

def linear_stack(df,axis=0):
    return np.sum(df,axis=axis)/df.shape[0]

def Nth_root_stack(df,N=4,axis=0):
    
    ned=np.abs(df)**(1/N)*np.sign(df)
    averaged=np.sum(ned,axis=axis)/df.shape[axis]
    
    return averaged**N*np.sign(averaged)
        
def pws_stack(df,v=2,smoothing=10,axis=0):
    
    if axis==0:
        
        trace=np.zeros(df.shape[1],dtype=complex)
        for i in range(df.shape[0]):
            h=signal.hilbert(df[i,:])
            trace+=h/np.abs(h)
        trace=np.abs(trace/df.shape[0])
        
    else:
        
        trace=np.zeros(df.shape[0],dtype=complex)
        for i in range(df.shape[1]):
            h=signal.hilbert(df[:,i])
            trace+=h/np.abs(h)
        trace=np.abs(trace/df.shape[1])
    
    operator=np.ones(smoothing)/smoothing
    trace=np.convolve(trace,operator,'same')
    pwsed=linear_stack(df,axis=axis)*trace**v
    
    return pwsed

def rms(data,axis=-1):
    
    return np.sqrt(np.sum(data**2,axis=axis)/data.shape[axis])


def fetch_sliced_image(arr,time,starting_time,
                       time_window=1,temporal_averageing=5,nstep=0.5,sampling_rate=20,
                       taper=None,normalize=True,masked=None):
    
    
    ### taper default to None, would be one type scipy.window function
    ### normalize default to True. Change to False to observe temporal evolution
    ### masked: values below this threshold will be masked
    
    
    winlen=int(time_window*sampling_rate)
    starting_point=max(int((starting_time+time-temporal_averageing/2)*sampling_rate),0)
    ending_point=min(starting_point+temporal_averageing*sampling_rate,arr.shape[-1])
    data=arr[:,:,starting_point:ending_point]
    stacked=np.zeros((arr.shape[0],arr.shape[1],winlen))
    
    c=0
    for start in range(0, data.shape[-1]-winlen+1,int(nstep*sampling_rate)):
        stacked+=data[:,:,start:start+winlen]
        c+=1

    if taper is not None:
        taper_func=signal.windows.get_window(taper,winlen)
        stacked=np.einsum('ijk,k->ijk',stacked,taper_func)
        
    if normalize:
        stacked/=np.max(np.abs(stacked))
    else:
        stacked/=c

    stacked=np.abs(stacked)
    
    if masked is not None:
        z=np.ma.masked_array(stacked,stacked<=masked)
        return np.abs(z)
    
    return stacked

def obspy_filter(data,lowcut,highcut,fs,order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    z, p, k = signal.iirfilter(order, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = signal.zpk2sos(z, p, k)
    new_data=signal.sosfilt(sos, data)
    return new_data

def locate_average(data,length,mode='maximum'):
    
    ave=np.convolve(data, np.ones(length)/length, mode='valid')
        
    if mode =='maximum':
        return np.max(ave)
    elif mode =='minimum':
        return np.min(ave)
    else: 
        raise 'error'

def snr(data,st,lt,fs):
    signal=locate_average(data,st*fs,'maximum')
    noise=locate_average(data,lt*fs,'minimum')
    snr=10*np.log10(signal/noise)
    return snr

def vec_corrcoef(X, y, axis=1):
    Xm = np.mean(X, axis=axis, keepdims=True)
    ym = np.mean(y)
    n = np.sum((X - Xm) * (y - ym), axis=axis)
    d = np.sqrt(np.sum((X - Xm)**2, axis=axis) * np.sum((y - ym)**2))
    return n / d

def cal_corr(cont,egf):
    
    swindow = np.lib.stride_tricks.sliding_window_view(cont, (len(egf),))
    corr = vec_corrcoef(swindow,egf)[:]
    
    return corr   