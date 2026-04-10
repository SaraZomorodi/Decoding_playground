import numpy as np
import os

class RAT:
    def __init__(self , n , lighting , filter_speed = True , t_max=1367.99):
        self.file_name = self.rat_retrieve(n)
        self.lighting = lighting
        self.t_max = t_max

        self.this_rat = self.opening_files()

        if filter_speed:
            speed = self.this_rat['speed']
            ind_ = np.where((speed < 100) & (speed > 3))
            self.X = self.this_rat['x'][ind_]
            self.Y = self.this_rat['y'][ind_]
            self.Z = self.this_rat['z'][ind_]
            self.HD = self.this_rat['hd'][ind_]
            self.grid_mod1 = self.this_rat['grid_mod1']
            self.grid_mod2 = self.this_rat['grid_mod2']
            self.grid_mod3 = self.this_rat['grid_mod3']
            self.V = self.this_rat['speed'][ind_]
            self.T = self.this_rat['t'][ind_]

        else:
            self.X = self.this_rat['x']
            self.Y = self.this_rat['y']
            self.Z = self.this_rat['z']
            self.HD = self.this_rat['hd']
            self.grid_mod1 = self.this_rat['grid_mod1']
            self.grid_mod2 = self.this_rat['grid_mod2']
            self.grid_mod3 = self.this_rat['grid_mod3']
            self.V = self.this_rat['speed']
            self.T = self.this_rat['t']

    def rat_retrieve(self , n ): 
        if n == 1: 
            file_name = 'moserlab_waaga_25843_2019-09-13_22-54-22_v1.npy'
        elif n == 2:
            file_name = 'moserlab_waaga_26018_2019-12-10_15-25-47_v1.npy'
        elif n == 3: 
            file_name = 'moserlab_waaga_26018_2019-12-14_16-03-44_v1.npy'
        elif n==4:
            file_name = 'moserlab_waaga_26718_2020-09-16_17-23-51_v1.npy'
        elif n==5: 
            file_name = 'moserlab_waaga_26820_2020-11-05_11-03-13_v1.npy'
        else: 
            raise " 404 rat not found"
        return file_name 
        
    def opening_files(self):
        dir_ = 'moser_dl'
        file_path =  os.path.join(dir_ , self.file_name)
        data = np.load(file_path , allow_pickle=True).item()
        if self.lighting=='light':
            section = 1
        elif self.lighting=='dark':
            section = 0

        # light = 1
        # dark = 0
        st = data["task"][section]["spike_timestamp"]
        cid = data["task"][section]["spike_cluster_id"]

        open_field_light_1 = data['task'][section]

        neurons_id = np.unique(open_field_light_1 ["spike_cluster_id"])
        module_id = data['module_id']

        this_rat = {}
        t = data['task'][section]['tracking']['t']
        t0 = t[0]

        this_rat['t'] = t-t[0]
        this_rat['x'] = data['task'][section]['tracking']['x']
        this_rat['y'] = data['task'][section]['tracking']['y']
        this_rat['z'] = data['task'][section]['tracking']['z']
        this_rat['hd'] = data['task'][section]['tracking']['hd']
        this_rat['grid_mod1'] = {int(i):st[cid == i]-t0  for i in neurons_id[module_id == 1]}
        this_rat['grid_mod2'] = {int(i):st[cid == i]-t0  for i in neurons_id[module_id == 2]}
        this_rat['grid_mod3'] = {int(i):st[cid == i]-t0  for i in neurons_id[module_id == 3]}
        dt = np.mean( this_rat['t'][1:] - this_rat['t'][:-1])
        speed = np.sqrt( ( this_rat['x'][1:] - this_rat['x'][:-1] )**2 + ( this_rat['y'][1:] - this_rat['y'][:-1] ) **2 )  / dt
        this_rat['speed'] = np.append([0] , speed) #assuming the motion is starting at speed 0


        if self.t_max is not None:
            # 1. Clip tracking data
            valid_t_idx = this_rat['t'] <= self.t_max
            
            this_rat['t'] = this_rat['t'][valid_t_idx]
            this_rat['x'] = this_rat['x'][valid_t_idx]
            this_rat['y'] = this_rat['y'][valid_t_idx]
            this_rat['z'] = this_rat['z'][valid_t_idx]
            this_rat['hd'] = this_rat['hd'][valid_t_idx]
            this_rat['speed'] = this_rat['speed'][valid_t_idx]

            # 2. Clip spike timestamps for all modules
            for mod in ['grid_mod1', 'grid_mod2', 'grid_mod3']:
                for cell_id, spikes in this_rat[mod].items():
                    # Filter array to only keep spikes that happen before or at t_max
                    this_rat[mod][cell_id] = spikes[spikes <= self.t_max]

        return this_rat
    
   
        
# def find_k(array , value):
#         """Finds index of the closest value in an array."""
#         return (np.abs(array-value)).argmin()
# def rate_map(x, y, t, margin ,spike, bin_width= 1):
#         """Computes the firing rate map using 2D histogram binning."""                             
#         x_edges = np.arange(min(x) - margin, max(x)+margin , bin_width)
#         y_edges = np.arange(min(y) - margin, max(y)+margin , bin_width)
#         ind_x = [find_k(t, i) for i in spike]
#         ind_y = [find_k(t, i) for i in spike]
#         occ = np.histogram2d(x, y, bins = (x_edges , y_edges))[0]
#         act = np.histogram2d(x[ind_x], y[ind_y], bins = (x_edges , y_edges))[0]
#         rate = act / (occ*0.02)
#         return rate