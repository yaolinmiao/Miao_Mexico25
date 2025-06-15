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
sys.path.append('/home/yaolinm/Projects/Mexico/')
from functions.mccc import *

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
        
# class reference_table:
    
#     def __init__(self,model,eq,wave='p'):
        
#         # reference table is a precalculated travel time table, to be interpolated
        
#         self.model=TauPyModel(model=model) # a taup velocity model
#         self.eq=eq # an earthquake object
#         self.wave=wave
        
#     def cal_table(self,min_dist,max_dist,stepsize):
        
#         # min_dist,max_dist and stepsize all in degree
        
#         distances=np.arange(min_dist,max_dist+stepsize,stepsize)
#         self.distances=distances
        
#         if self.wave not in ['p','P','s','S']:
#             raise Exception('Invalid wave type')

#         travel_times=[]

#         for distance in distances:

#             try:
#                 arrivals=self.model.get_travel_times(source_depth_in_km=self.eq.depth,
#                                             distance_in_degree=distance,
#                                             phase_list=[self.wave])
#                 travel_times.append(round(arrivals[0].time,6))

#             except Exception as e:
#                 print(e)
#                 raise 
        
#         self.travel_times=np.array(travel_times)
        

# class sensor:
    
#     def __init__(self,longitude,latitude,
#                  trace_len=4000,sampling=50,eq_start=40,
#                  type='seismometer'):
        
#         self.longitude=longitude
#         self.latitude=latitude
#         self.trace_len=trace_len
#         self.sampling=sampling
#         self.eq_start=eq_start
        
#         if type=='seismometer':
#             self.type=1
#         elif type=='das':
#             self.type=0
#         else:
#             raise Exception('Invalid sensor type')
            
#     def add_resp_trace(self,window_len,amplitude,roll_index,window_type='triang'):
        
#         tr=np.zeros(self.trace_len)
#         tr[self.eq_start*self.sampling:self.eq_start*self.sampling+window_len*self.sampling+1]=signal.get_window(window_type,window_len*self.sampling+1)
#         tr=np.roll(tr,roll_index)
#         self.resp_trace=tr
        
#     def add_seis_trace(self,seis_trace):
        
#         assert len(seis_trace)==self.trace_len, 'Unequal trace length'
#         self.seis_trace=seis_trace

#     def calculate_ttime(self,grid,rtable):
        
#         ### grid is a gridize class object
#         ### rtable is a reference_table object
        
#         distances=rtable.distances
#         traveltimes=rtable.travel_times
#         arrival_mat=np.zeros((len(grid.longitude_grids),len(grid.latitude_grids)))
        
#         f=interpolate.interp1d(distances,traveltimes)
        
#         for long_index in range(len(grid.longitude_grids)):
#             long=grid.longitude_grids[long_index]

#             for lati_index in range(len(grid.latitude_grids)):
#                 lati=grid.latitude_grids[lati_index]

#                 dist=round(locations2degrees(self.latitude,self.longitude,lati,long),6)

#                 try:
#                     temp_traveltime=f([dist])[0]
#                 except:
#                     temp_traveltime=dist/distances[-1]*traveltimes[-1]
#                 arrival_mat[long_index,lati_index]=temp_traveltime
            
#         self.arrival_mat=arrival_mat
        
#     def cal_epitime(self,eq,rtable):
        
#         ### eq is a earthquake class object
#         ### rtable is a reference_table object
        
#         distances=rtable.distances
#         traveltimes=rtable.travel_times
#         f=interpolate.interp1d(distances, traveltimes)
#         dist=round(locations2degrees(self.latitude,self.longitude,eq.latitude,eq.longitude),6)
#         self.epitime=f([dist])[0]
        
# class bp_bmfm:
    
#     def __init__(self,sensor_list,rtable,grid,eq):
        
#         ## backprojection with conventional time domain beamforming
        
#         assert len(sensor_list)>1, 'Need more than one sensor'
        
#         sensor_types=[]
#         for sensor in sensor_list:
#             sensor_types.append(sensor.type)
#         if len(set(sensor_types))==1:
#             self.bp_type='beam power'
#         else:
#             self.bp_type='pdf'
        
#         self.sensor_list=sensor_list ## sensor list is a list of sensor objects, in good order
#         self.rtable=rtable ## reference table object
#         self.grid=grid ## gridize object
#         self.eq=eq ## earthquake object
        
#     def get_response(self,nskips=100,ref=None):
        
#         x_mesh,y_mesh=self.grid.mesh()
#         syn_master=np.zeros((x_mesh.shape[1],x_mesh.shape[0],self.sensor_list[0].trace_len))
        
#         if ref is None:
#             ref=np.min(self.sensor_list[0].arrival_mat)
            
#         if isinstance(ref, list):
        
#             for i in range(syn_master.shape[0]):
#                 for j in range(syn_master.shape[1]):  
#                     for k,one_sensor in enumerate (self.sensor_list[::nskips]):
#                         rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref[k])*one_sensor.sampling).astype('int')
#                         rolled=np.roll(one_sensor.resp_trace,rolling_index)
#                         syn_master[i,j,:]+=rolled
        
#         else:
#             for i in range(syn_master.shape[0]):
#                 for j in range(syn_master.shape[1]):  
#                     for one_sensor in self.sensor_list[::nskips]:
#                         rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref)*one_sensor.sampling).astype('int')
#                         rolled=np.roll(one_sensor.resp_trace,-rolling_index)
#                         syn_master[i,j,:]+=rolled
        
#         sigma=np.max(syn_master)/np.mean(np.max(syn_master,axis=-1))
            
#         self.syn_master=syn_master
#         self.sigma=sigma
    
#     def run_bp(self,ref=None):
           
#         x_mesh,y_mesh=self.grid.mesh()
#         bp_image=np.zeros((x_mesh.shape[1],x_mesh.shape[0],self.sensor_list[0].trace_len))
        
#         if ref is None:
#             ref=np.min(self.sensor_list[0].arrival_mat)
            
#         if isinstance(ref, list):
        
#             for i in range(bp_image.shape[0]):
#                 for j in range(bp_image.shape[1]):  
#                     for k,one_sensor in enumerate (self.sensor_list):
#                         rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref[k])*one_sensor.sampling).astype('int')
#                         rolled=np.roll(one_sensor.seis_trace,rolling_index)
#                         rolled/=np.max(np.abs(rolled))
#                         bp_image[i,j,:]+=np.abs(rolled)**(1/4)*np.sign(rolled)      
                    
# #                         bp_image[i,j,:]+=rolled
#         else:
#             for i in range(bp_image.shape[0]):
#                 for j in range(bp_image.shape[1]):  
#                     for one_sensor in self.sensor_list:
#                         rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref)*one_sensor.sampling).astype('int')
#                         rolled=np.roll(one_sensor.seis_trace,-rolling_index)
#                         rolled/=np.max(np.abs(rolled))
#                         bp_image[i,j,:]+=np.abs(rolled)**(1/4)*np.sign(rolled)     

#         bp_image/=len(self.sensor_list)
#         self.bp_image=bp_image**4*np.sign(bp_image)
        
#     def plot_response(self):
        
#         x_mesh,y_mesh=self.grid.mesh()
#         syn_master_max=np.max(self.syn_master,axis=-1)
        
#         fig=plt.figure(dpi=200)
#         pc=plt.pcolormesh(x_mesh,y_mesh,syn_master_max.T)
#         plt.scatter(self.eq.longitude,self.eq.latitude,marker='*',c='r',s=100)
#         plt.title('sigma of {}'.format(str(self.sigma)))
#         fig.colorbar(pc)
#         plt.show()
        
        
#     def plot_bp(self):
        
#         x_mesh,y_mesh=self.grid.mesh()
#         bp_image_max=np.max(self.bp_image,axis=-1)
        
#         fig=plt.figure(dpi=200)
#         pc=plt.pcolormesh(x_mesh,y_mesh,bp_image_max.T)
#         plt.scatter(self.eq.longitude,self.eq.latitude,marker='*',c='r',s=100)
#         plt.colorbar(pc)
#         plt.show()
        
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
                 trace_len=4000,sampling=50,eq_start=40,
                 type='seismometer'):
        
        self.longitude=longitude
        self.latitude=latitude
        self.trace_len=trace_len
        self.sampling=sampling
        self.eq_start=eq_start
        self.epitime=[]
        
        if type=='seismometer':
            self.type=1
        elif type=='das':
            self.type=0
        else:
            raise Exception('Invalid sensor type')
            
    def add_resp_trace(self,window_len,amplitude,roll_index,window_type='triang'):
        
        tr=np.zeros(self.trace_len)
        tr[self.eq_start*self.sampling:self.eq_start*self.sampling+window_len*self.sampling+1]=signal.get_window(window_type,window_len*self.sampling+1)
        tr=np.roll(tr,roll_index)
        self.resp_trace=tr
        
    def add_seis_trace(self,seis_trace,amplitude=1):
        
        assert len(seis_trace)==self.trace_len, 'Unequal trace length'
        self.seis_trace=np.array(seis_trace)*amplitude

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
        
    def cal_epitime(self,eq,rtable):
        
        ### eq is a earthquake class object
        ### rtable is a reference_table object
        
        distances=rtable.distances
        traveltimes=rtable.travel_times
        f=interpolate.interp1d(distances, traveltimes)
        dist=round(locations2degrees(self.latitude,self.longitude,eq.latitude,eq.longitude),6)
        self.epitime.append(f([dist])[0])
        
class bp_bmfm:
    
    def __init__(self,sensor_list,rtable,grid,eq):
        
        ## backprojection with conventional time domain beamforming
        
        assert len(sensor_list)>1, 'Need more than one sensor'
        
        sensor_types=[]
        for sensor in sensor_list:
            sensor_types.append(sensor.type)
        if len(set(sensor_types))==1:
            self.bp_type='beam power'
        else:
            self.bp_type='pdf'
        
        self.sensor_list=sensor_list ## sensor list is a list of sensor objects, in good order
        self.rtable=rtable ## reference table object
        self.grid=grid ## gridize object
        self.eq=eq ## earthquake object
        
    def get_response(self,nskips=100,ref=None):
        
        x_mesh,y_mesh=self.grid.mesh()
        syn_master=np.zeros((x_mesh.shape[1],x_mesh.shape[0],self.sensor_list[0].trace_len))
        
        if ref is None:
            ref=np.min(self.sensor_list[0].arrival_mat)
            
        if isinstance(ref, list):
        
            for i in range(syn_master.shape[0]):
                for j in range(syn_master.shape[1]):  
                    for k,one_sensor in enumerate (self.sensor_list[::nskips]):
                        rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref[k])*one_sensor.sampling).astype('int')
                        rolled=np.roll(one_sensor.resp_trace,-rolling_index)
                        syn_master[i,j,:]+=rolled
        
        else:
            for i in range(syn_master.shape[0]):
                for j in range(syn_master.shape[1]):  
                    for one_sensor in self.sensor_list[::nskips]:
                        rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref)*one_sensor.sampling).astype('int')
                        rolled=np.roll(one_sensor.resp_trace,-rolling_index)
                        syn_master[i,j,:]+=rolled
        
        sigma=np.max(syn_master)/np.mean(np.max(syn_master,axis=-1))
            
        self.syn_master=syn_master
        self.sigma=sigma
    
    def run_bp(self,ref=None):
           
        x_mesh,y_mesh=self.grid.mesh()
        bp_image=np.zeros((x_mesh.shape[1],x_mesh.shape[0],self.sensor_list[0].trace_len))
        
        if ref is None:
            ref=np.min(self.sensor_list[0].arrival_mat)
            
        if isinstance(ref, list):
        
            for i in range(bp_image.shape[0]):
                for j in range(bp_image.shape[1]):  
                    for k,one_sensor in enumerate (self.sensor_list):
                        rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref[k])*one_sensor.sampling).astype('int')
                        rolled=np.roll(one_sensor.seis_trace,-rolling_index)
                        rolled/=np.max(np.abs(rolled))
                        bp_image[i,j,:]+=np.abs(rolled)**(1/4)*np.sign(rolled)      
                        
        else:
            for i in range(bp_image.shape[0]):
                for j in range(bp_image.shape[1]):  
                    for one_sensor in self.sensor_list:
                        rolling_index=np.rint((one_sensor.arrival_mat[i,j]-ref)*one_sensor.sampling).astype('int')
                        rolled=np.roll(one_sensor.seis_trace,-rolling_index)
                        rolled/=np.max(np.abs(rolled))
                        bp_image[i,j,:]+=np.abs(rolled)**(1/4)*np.sign(rolled)     

        bp_image/=len(self.sensor_list)
        self.bp_image=bp_image**4*np.sign(bp_image)
        
    def plot_response(self):
        
        x_mesh,y_mesh=self.grid.mesh()
        syn_master_max=np.max(self.syn_master,axis=-1)
        
        fig=plt.figure(dpi=200)
        pc=plt.pcolormesh(x_mesh,y_mesh,syn_master_max.T)
        plt.scatter(self.eq.longitude,self.eq.latitude,marker='*',c='r',s=100)
        plt.title('sigma of {}'.format(str(self.sigma)))
        fig.colorbar(pc)
        plt.show()
        
    def plot_bp(self):
        
        x_mesh,y_mesh=self.grid.mesh()
        bp_image_max=np.max(self.bp_image,axis=-1)
        
        fig=plt.figure(dpi=200)
        pc=plt.pcolormesh(x_mesh,y_mesh,bp_image_max.T)
        plt.scatter(self.eq.longitude,self.eq.latitude,marker='*',c='r',s=100)
        plt.colorbar(pc)
        plt.show()
        
def fetch_sliced_image(arr,time,starting_time,
                       time_window=1,temporal_averageing=5,sampling_rate=20,
                       taper=None,normalize=True,masked=None):
    
    
    ### taper default to None, would be one type scipy.window function
    ### normalize default to True. Change to False to observe temporal evolution
    ### masked: values below this threshold will be masked
    
    
    nwindows=temporal_averageing//time_window
    winlen=int(time_window*sampling_rate)
    
    starting_point=int((starting_time+time-time_window/2-(temporal_averageing-1)/2)*sampling_rate)
    stacked=np.zeros((arr.shape[0],arr.shape[1],winlen))
        
    for i in range(nwindows):
        stacked+=arr[:,:,starting_point+i*winlen:starting_point+(i+1)*winlen]
        
    if taper is not None:
        taper_func=signal.windows.get_window(taper,winlen)
        stacked=np.einsum('ijk,k->ijk',stacked,taper_func)
        
    if normalize:
        stacked/=np.max(np.abs(stacked))
    else:
        stacked/=nwindows
        stacked/=np.max(np.abs(arr))
    
    stacked=np.abs(stacked)
    
    if masked is not None:
        z=np.ma.masked_array(stacked,stacked<=masked)
        return np.abs(z)
    
    return stacked

def normalize(data):
    return data/np.max(np.abs(data))

def rms(data,axis=-1):
    
    return np.sqrt(np.sum(data**2,axis=axis)/data.shape[axis])

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