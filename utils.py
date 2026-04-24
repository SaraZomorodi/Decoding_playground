import numpy as np
import os
from scipy.ndimage import gaussian_filter1d, gaussian_filter


class RAT:
    def __init__(self , n , lighting , filter_speed = True , t_max=None , sigma_pos = 2.0 , min_v=3.0, max_v=100.0):
        self.file_name = self.rat_retrieve(n)
        self.lighting = lighting
        self.t_max = t_max
        self.sigma_pos = sigma_pos
        self.min_v = min_v
        self.max_v = max_v

        self.this_rat = self.opening_files()
        # 1. Smooth X and Y coordinates
        x_smooth = gaussian_filter1d(self.this_rat['x'], sigma=self.sigma_pos)
        y_smooth = gaussian_filter1d(self.this_rat['y'], sigma=self.sigma_pos)
        if filter_speed:
            # speed = self.this_rat['speed']
            
            
            # 2. Calculate continuous velocity using smoothed positions & time steps
            # dx = np.gradient(x_smooth, self.this_rat['t'])
            # dy = np.gradient(y_smooth, self.this_rat['t'])
            # dt = np.gradient(self.this_rat['t'])
            # self.V = np.sqrt(dx**2 + dy**2) / dt


            dt_array = np.gradient(self.this_rat['t'])
            vx = np.gradient(x_smooth) / dt_array
            vy = np.gradient(y_smooth) / dt_array
            self.V = np.sqrt(vx**2 + vy**2)


            ind_ = np.where((self.V < self.max_v) & (self.V > self.min_v))

            self.X = x_smooth[ind_]
            self.Y = y_smooth[ind_]
            self.Z = self.this_rat['z'][ind_]
            self.HD = self.this_rat['hd'][ind_]
            self.grid_mod1 = self._filter_spikes_by_speed(self.this_rat['grid_mod1'])
            self.grid_mod2 = self._filter_spikes_by_speed(self.this_rat['grid_mod2'])
            self.grid_mod3 = self._filter_spikes_by_speed(self.this_rat['grid_mod3'])
            
            self.ind_ = ind_
            
            self.valid_mask = (self.V >= self.min_v) & (self.V <= self.max_v)
            
            self.V = self.V[ind_]
            change_indices = np.where(np.diff(self.valid_mask.astype(int)) == -1)[0]
            self.removed_starts = self.this_rat['t'][change_indices]

            # self.V = self.this_rat['speed'][ind_]
            self.T = self.this_rat['t'][ind_]

            # Calculate the original sampling rate (dt)
            dt_original = np.mean(np.diff(self.this_rat['t']))
            
            # Create a continuous time array: 0, dt, 2*dt, ..., N*dt
            # The length of this array will perfectly match len(self.X)
            self.t_active = np.arange(len(self.X)) * dt_original

            

            # ---------------------------------------------------------
            # 5. Create continuous t_active
            dt_original = np.mean(np.diff(self.this_rat['t']))
            self.t_active = np.arange(len(self.X)) * dt_original

            # ---------------------------------------------------------
            # NEW: Map the end of invalid intervals back to t_active
            t_orig = self.this_rat['t']
            
            # Pad the mask with True at both ends to easily catch intervals 
            # that start at frame 0 or end at the very last frame.
            padded_mask = np.concatenate(([True], self.valid_mask, [True]))
            diffs = np.diff(padded_mask.astype(int))
            
            # -1 means True -> False (Start of invalid interval)
            invalid_starts = np.where(diffs == -1)[0]
            # 1 means False -> True (End of invalid interval / Resume point)
            invalid_ends = np.where(diffs == 1)[0]
            
            # Safety catch: if the very last frame of the recording is invalid,
            # clip the end index so we don't go out of bounds of t_orig
            invalid_ends = np.clip(invalid_ends, 0, len(t_orig) - 1)
            
            # 1. Get the original time (self.this_rat['t']) at the end of the interval
            self.end_times_orig = t_orig[invalid_ends]
            
            # 2. Calculate the duration of each out-of-threshold interval
            invalid_durations = t_orig[invalid_ends] - t_orig[invalid_starts]
            
            # 3. Calculate the total duration of removed intervals up to that moment
            self.cumulative_invalid_durations = np.cumsum(invalid_durations)
            
            # 4. Subtract the cumulative duration from the original end times
            # This perfectly maps the "resume" points onto your t_active timeline!
            self.mapped_resume_times = self.end_times_orig - self.cumulative_invalid_durations
            # ---------------------------------------------------------


        else:
            self.X = self.this_rat['x']
            self.Y = self.this_rat['y']
            self.Z = self.this_rat['z']
            self.HD = self.this_rat['hd']
            self.grid_mod1 = self.this_rat['grid_mod1']
            self.grid_mod2 = self.this_rat['grid_mod2']
            self.grid_mod3 = self.this_rat['grid_mod3']
            dx = np.gradient(x_smooth, self.this_rat['t'])
            dy = np.gradient(y_smooth, self.this_rat['t'])
            dt = np.gradient(self.this_rat['t'])
            self.V = np.sqrt(dx**2 + dy**2) / dt
            self.T = self.this_rat['t']


            dt_array = np.gradient(self.this_rat['t'])
            vx = np.gradient(x_smooth) / dt_array
            vy = np.gradient(y_smooth) / dt_array
            self.V_full = np.sqrt(vx**2 + vy**2)

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
    
    def _filter_spikes_by_speed(self , spikes_mod):
        """Removes any spike timestamps that occurred during invalid speeds."""
        new_spikes_mod = {}
        for cell_id, spikes in spikes_mod.items():
            spikes = np.asarray(spikes, dtype=float)
            if len(spikes) == 0:
                new_spikes_mod[cell_id] = spikes
                continue
            # Restrict to tracked recording window
            valid_time = (spikes >= self.this_rat['t'][0]) & (spikes <= self.this_rat['t'][-1])
            # spikes = spikes[valid_time]
            
            # Find the speed at the time the spike fired
            idx = np.searchsorted(self.this_rat['t'], spikes)
            idx = np.clip(idx, 0, len(self.this_rat['t']) - 1)

            speeds_at_spikes = self.V[idx]

            # Apply strict [3, 100] speed bounds
            valid_speed = (speeds_at_spikes >= self.min_v) & (speeds_at_spikes <= self.max_v)
            new_spikes_mod[cell_id] = spikes[valid_speed]
        return new_spikes_mod
    
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