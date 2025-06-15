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
from funcs.mccc import *
from funcs.utils import *

from numba import njit
import numba

class earthquake:
    
    def __init__(self,longitude,latitude,depth):
        
        self.longitude=longitude
        self.latitude=latitude
        self.depth=depth
        
class gridize:
    
    def __init__(self,longitude_range,latitude_range,stepsize=0.025):
        
        self.longitude_range=longitude_range
        self.latitude_range=latitude_range
        self.stepsize=stepsize
        self.longitude_grids=np.arange(longitude_range[0],longitude_range[1],self.stepsize)
        self.latitude_grids=np.arange(latitude_range[0],latitude_range[1],self.stepsize)
        
    def mesh(self):
        
        x_mesh_new,y_mesh_new=np.meshgrid(self.longitude_grids,self.latitude_grids)
        return x_mesh_new,y_mesh_new
        
class reference_table:
    
    def __init__(self,model,eq,wave='p'):
        
        # reference table is a precalculated travel time table, to be interpolated
        
        self.model=TauPyModel(model=model) # a taup velocity model
        self.eq=eq # an earthquake object
        self.wave=wave
        
    def cal_table(self,min_dist,max_dist,stepsize):
        
        # min_dist,max_dist and stepsize all in degree
        
        distances=np.arange(min_dist,max_dist+stepsize,stepsize)
        self.distances=distances
        
        if self.wave not in ['p','P','s','S']:
            raise Exception('Invalid wave type')

        travel_times=[]

        for distance in distances:

            
            arrivals=self.model.get_travel_times(source_depth_in_km=self.eq.depth,
                                        distance_in_degree=distance,phase_list=[self.wave])

            try:
                travel_times.append(round(arrivals[0].time,6))

            except Exception as e:
                print(e)
                travel_times.append(None)
                 
        
        self.travel_times=np.array(travel_times)
        

class sensor:
    
    def __init__(self,longitude,latitude,
                 trace_len=4000,sampling=50,eq_start=40):
        
        self.longitude=longitude
        self.latitude=latitude
        self.trace_len=trace_len
        self.sampling=sampling
        self.eq_start=eq_start
        self.epitime=None
        self.subevents=[]
        
    def add_seis_trace(self,seis_trace,amplitude=1):
        
        assert len(seis_trace)==self.trace_len, 'Unequal trace length'
        self.trace=normalize(np.array(seis_trace))*amplitude
        
    def cal_epitime(self,eq,rtable,etype='mainevent'):
        
        ### eq is a earthquake class object
        ### rtable is a reference_table object
        
        distances=rtable.distances
        traveltimes=rtable.travel_times
        f=interpolate.interp1d(distances, traveltimes)
        dist=round(locations2degrees(self.latitude,self.longitude,eq.latitude,eq.longitude),6)
        if etype=='mainevent':
            self.epitime=f([dist])[0]
        elif etype=='subevent':
            self.subevents.append(f([dist])[0])
        else:
            raise Exception('Invalid sensor type')
    
    def calculate_ttime(self,grid,rtable):
        
        ### grid is a gridize class object
        ### rtable is a reference_table object
        
        distances=rtable.distances
        traveltimes=rtable.travel_times
        arrival_mat=np.zeros((len(grid.longitude_grids),len(grid.latitude_grids)))
        
        f=interpolate.interp1d(distances,traveltimes)
        
        for long_index in range(len(grid.longitude_grids)):
            long=grid.longitude_grids[long_index]

            for lati_index in range(len(grid.latitude_grids)):
                lati=grid.latitude_grids[lati_index]

                dist=round(locations2degrees(self.latitude,self.longitude,lati,long),6)

                try:
                    temp_traveltime=f([dist])[0]
                except:
                    temp_traveltime=dist/distances[-1]*traveltimes[-1]
                arrival_mat[long_index,lati_index]=temp_traveltime
            
        self.arrival_mat=arrival_mat
        self.shift_mat=np.rint((self.arrival_mat-self.epitime)*self.sampling).astype('int')

@numba.njit(parallel=True)
def apply_shift_numba(seismogram, shift_matrix):
    rows, cols = shift_matrix.shape
    shifted_traces = np.empty((rows, cols, len(seismogram)), dtype=seismogram.dtype)

    for i in numba.prange(rows):  # Parallel loop
        for j in range(cols):
            shifted_traces[i, j] = np.roll(seismogram, shift_matrix[i, j])

    return shifted_traces

@numba.njit(parallel=True)
def process_das_sensors(traces, shift_matrices, stacking=4):

    signal_length = traces.shape[1]
    num_channels, rows, cols = shift_matrices.shape
    master = np.empty((rows, cols,signal_length), dtype=traces.dtype)

    for i in numba.prange(num_channels):
        shifted = apply_shift_numba(traces[i], -shift_matrices[i])
        transformed = np.sign(shifted) * (np.abs(shifted) ** (1/stacking))
        master += transformed
    
    master /= num_channels
    master = master**stacking*np.sign(master)
    return master

class bp_bmfm:
    
    def __init__(self,sensor_list,rtable,grid,eq):
        
        ## backprojection with conventional time domain beamforming
        
        assert len(sensor_list)>1, 'Need more than one sensor'
  
        self.sensor_list=sensor_list ## sensor list is a list of sensor objects, in good order
        self.rtable=rtable ## reference table object
        self.grid=grid ## gridize object
        self.eq=eq ## earthquake object
        x_mesh,y_mesh=self.grid.mesh()
        self.master=np.zeros((x_mesh.shape[1],x_mesh.shape[0],self.sensor_list[0].trace_len))
        

#     @njit(parallel=True)
#     def run_bp(self,nskips=1,stacking=4):
            
#         if not isinstance(stacking, int) :
#             raise TypeError('stacking has to be an integer')
        
#         master=self.master
   
#         for i in range(master.shape[0]):
#             for j in range(master.shape[1]):  
#                 for k,one_sensor in enumerate (self.sensor_list[::nskips]):
#                     rolled=np.roll(one_sensor.trace,-one_sensor.shift_mat[i,j])
#                     master[i,j,:]+=np.abs(rolled)**(1/stacking)*np.sign(rolled)    
#         master/=len(self.sensor_list[::nskips])
#         master=master**stacking*np.sign(master)
            
#         sigma=np.max(master)/np.mean(np.max(master,axis=-1))
            
#         self.master=master
#         self.sigma=sigma

    def run_bp(self,nskips=1,stacking=4):
            
        if not isinstance(stacking, int) :
            raise TypeError('stacking has to be an integer')
        
  
        traces = np.array([channel.trace for channel in self.sensor_list[::nskips]])
        shift_matrices = np.array([channel.shift_mat for channel in self.sensor_list[::nskips]]) 
   
        master=process_das_sensors(traces, shift_matrices)
            
        sigma=np.max(master)/np.mean(np.max(master,axis=-1))
            
        self.master=master
        self.sigma=sigma
        
    def _apply_shift(seismogram, shift_matrix):
        rows, cols = shift_matrix.shape
        shifted_traces = np.empty((rows, cols, len(seismogram)), dtype=seismogram.dtype)

        for i in range(rows):
            for j in range(cols):
                shifted_traces[i, j] = np.roll(seismogram, shift_matrix[i, j])

        return shifted_traces
    
        
    def plot_bp(self):
        
        x_mesh,y_mesh=self.grid.mesh()
        master=np.max(self.master,axis=-1)
        
        fig=plt.figure(dpi=200)
        pc=plt.pcolormesh(x_mesh,y_mesh,master.T)
        plt.scatter(self.eq.longitude,self.eq.latitude,marker='*',c='r',s=100)
        plt.colorbar(pc)
        plt.show()
