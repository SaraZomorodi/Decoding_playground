import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, rotate
import tqdm
import traceback
import csv
import matplotlib.animation as animation

class TuningCurves:
    def __init__(self , rat_ , filter_speed, speed_threshold , res , Lim , margin , fold = 1 ,cv = 0):
        self.speed_threshold = speed_threshold #diluting spikes trains  - but according to speed thresohld
        if filter_speed:
            self.rat = self.filter_speed_all(rat_)
        else:
            self.rat = rat_
        self.res = res
        self.x_traj = np.round(self.rat['x'], self.res) # x location of the animal's trajectory [cm] rounded to 1cm resulotion
        self.y_traj = np.round(self.rat['y'], self.res) # y location of the animal's trajectory [cm] rounded to 1cm resulotion
        self.time   = self.rat['t'] # corresponding time stamps (s) , starting from zero 
        self.dr = 10**self.res # spatial resulotion (cm)
        self.Lim = Lim #radius of the arana (cm)
        self.margin= margin # (cm) margin addition to the radius for not losing (and reflecting) probability leaks from outside the real arena back in (Must be as in multi exp deocded)

        self.cv = cv # should be 0 in case no cv
        if cv > 0:
            n = len(self.rat['t'])//cv
            self.t_start_mask = self.rat['t'][(fold - 1) * n]
            self.t_end_mask   = self.rat['t'][ fold      * n]
        self.tuning_curve = self.get_all_tc()



    def find_k(self, array , value):
        """Finds index of the closest value in an array."""
        return (np.abs(array-value)).argmin()

    def rate_map(self, x, y, t, margin ,spike, bin_width= 1):
        """Computes the firing rate map using 2D histogram binning."""
        if self.cv > 0:
            keep = (spike <= self.t_start_mask) | (spike >= self.t_end_mask)
            spike = spike[keep]
            # spike[np.where ((spike > self.t_start_mask) & (spike < self.t_end_mask) )] = 0 #wrong approach. turns them to 0 and still contibutes to the decoding
                                      
        x_edges = np.arange(min(x) - margin, max(x)+margin , bin_width)
        y_edges = np.arange(min(y) - margin, max(y)+margin , bin_width)
        ind_x = [self.find_k(t, i) for i in spike]
        ind_y = [self.find_k(t, i) for i in spike]
        occ = np.histogram2d(x, y, bins = (x_edges , y_edges))[0]
        act = np.histogram2d(x[ind_x], y[ind_y], bins = (x_edges , y_edges))[0]
        rate = act / (occ*0.02)
        return rate

    def get_tuning_curve(self, mod, bin_width):
        """Computes tuning curves for the grid module mod"""
        tc_dict = {}
        print(f"Computing tuning curves for {mod}...")
        for k in tqdm.tqdm(self.rat[mod].keys(), desc=f"Tuning curves ({mod})" , miniters=10):
      
        # for i , k in enumerate(self.rat[mod].keys()):
            tc  = self.rate_map(self.x_traj, self.y_traj, self.time, self.margin ,self.rat[mod][k], bin_width= bin_width) 
            p_tc = tc / np.nansum(tc)
            # tc_arr[i] = tc
            tc_dict[k] = p_tc + 10e-5
        return tc_dict

    def get_all_tc(self):
        mod1_tc = self.get_tuning_curve('grid_mod1', bin_width=1)
        mod2_tc = self.get_tuning_curve('grid_mod2', bin_width=1)
        try:
            mod3_tc = self.get_tuning_curve('grid_mod3', bin_width=1)
        except:
            pass
        all_tc = mod1_tc
        all_tc.update(mod2_tc)
        try:
            all_tc.update(mod3_tc)
        except:
            pass
        return all_tc
    



class KernelDecoder:
    """
    Class to perform Bayesian position decoding using the kernel-based method
    from spike trains recorded in acircular arena from rats HPC
    """
    def __init__(self , rat_ , all_tc , kernel_type, filter_speed,speed_threshold , res , Lim , margin , Rel_Tau , Tau , fold = 1 ,cv = 0):
        self.speed_threshold = speed_threshold #diluting spikes trains  - but according to speed thresohld
        if filter_speed:
            self.rat = self.filter_speed_all(rat_)
        else:
            self.rat = rat_
        self.res = res
        self.x_traj = np.round(self.rat['x'], self.res) # x location of the animal's trajectory [cm] rounded to 1cm resulotion
        self.y_traj = np.round(self.rat['y'], self.res) # y location of the animal's trajectory [cm] rounded to 1cm resulotion
        self.time   = self.rat['t'] # corresponding time stamps (s) , starting from zero 
        self.dr = 10**self.res # spatial resulotion (cm)
        self.Lim = Lim #radius of the arana (cm)
        self.margin= margin # (cm) margin addition to the radius for not losing (and reflecting) probability leaks from outside the real arena back in (Must be as in multi exp deocded)

        self.x = np.arange(-Lim - margin, Lim + margin + self.dr, self.dr) # x projection of the arena
        self.L = self.x.shape[0] #shape of the arena
        self.X, self.Y = np.meshgrid(self.x, self.x)
        self.R_sq = self.X**2 + self.Y**2 # sqr distance from the origin (cm^2)
        self.out_x , self.out_y  = np.where(self.R_sq > Lim**2) # indx outside of areana
        self.dt = np.mean(self.time[1:] - self.time[:-1])
        self.num_points = self.time.shape[0]

        self.arena_shape_x =  np.arange(min(self.x_traj) - margin, max(self.x_traj)+margin , 1).shape[0] - 1
        self.arena_shape_y =  np.arange(min(self.y_traj) - margin, max(self.y_traj)+margin , 1).shape[0] - 1

        # exp Kernel
        self.Rel_Tau= Rel_Tau #how many Tau's back is it relevant to look at on the spike trains? 
        self.Tau = Tau # (s) Time constatnt
        self.Tau_indicies = np.round(self.Tau / self.dt) # How many time index
        self.Rel_times = self.Rel_Tau * self.Tau_indicies # Number of reletaive time points to look back 

        self.kernel_type = kernel_type
        
        self.all_tc = all_tc
        self.cv = cv # should be 0 in case no cv
        if cv > 0:
            n = len(self.rat['t'])//cv
            self.ind_start_mask = (fold - 1) * n
            self.ind_end_mask   =  fold      * n



    def find_k(self, array , value):
        """Finds index of the closest value in an array."""
        return (np.abs(array-value)).argmin()
    
    def filter_speed_all(self, rat):
        """Filters spikes and trajectory data based on speed threshold."""
        for mod in ['grid_mod1' , 'grid_mod2' , 'grid_mod3']:
            try:
                for k in rat[mod].keys():
                    spike = rat[mod][k]
                    filtered_spikes = [s for s in spike if rat['speed'][self.find_k(rat['t'], s)] >= self.speed_threshold]
                    rat[mod][k] = np.array(filtered_spikes)
            except:
                pass
        ind_ = np.where(rat['speed'] >= self.speed_threshold)
        rat['t'] = rat['t'][ind_]
        rat['x'] = rat['x'][ind_]
        rat['y'] = rat['y'][ind_]
        rat['speed'] = rat['speed'][ind_]
        return rat


    def pad_rate_map(self , rate):
        """Pads rate map to fit a target shape based on arena limits."""
        target_shape=(self.Lim*2, self.Lim*2)
        current_shape = rate.shape
        padding_x = (target_shape[0] - current_shape[0]) // 2
        padding_y = (target_shape[1] - current_shape[1]) // 2
        # target shape  odd
        pad_x = (padding_x, target_shape[0] - current_shape[0] - padding_x)
        pad_y = (padding_y, target_shape[1] - current_shape[1] - padding_y)
        padded_rate = np.pad(rate, (pad_x, pad_y), mode='constant', constant_values=0)
        return padded_rate

    def mask_outside_circle(self, rate):
        """Masks areas outside the circular arena in the rate map."""
        rows, cols = rate.shape
        center_x, center_y = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = self.Lim + self.margin
        mask = distance_from_center <= r
        rate_masked = rate.astype(float) 
        rate_masked[~mask] = np.nan
        return rate_masked
    
    def rate_map(self, x, y, t, margin ,spike, bin_width= 1):
        """Computes the firing rate map using 2D histogram binning."""
        x_edges = np.arange(min(x) - margin, max(x)+margin , bin_width)
        y_edges = np.arange(min(y) - margin, max(y)+margin , bin_width)
        ind_x = [self.find_k(t, i) for i in spike]
        ind_y = [self.find_k(t, i) for i in spike]
        occ = np.histogram2d(x, y, bins = (x_edges , y_edges))[0]
        act = np.histogram2d(x[ind_x], y[ind_y], bins = (x_edges , y_edges))[0]
        rate = act / (occ*0.02)
        return rate
    

    def get_spikes(self, t1 , t2 , mod , selected_IDs = False ): # returns spike times for neurons spiking between t1 and t2
        """Returns spike times for neurons firing between t1 and t2."""
        try:
            t1_swap, t2_swap = min(t1, t2), max(t1, t2)
            spike_dict = {} # dict contains all relevant spikes - assuming no overlap between cell IDs
            if not selected_IDs:
                for k in self.rat[mod].keys():
                    spike = self.rat[mod][k]        
                    indx_ = np.where( (spike <= t2_swap) & (spike > t1_swap))[0]
                    if indx_.shape[0] != 0:
                        spike_dict[k] = spike[indx_]
            if selected_IDs:
                for k in selected_IDs: 
                        if k in self.rat[mod].keys():
                            spike = self.rat[mod][k]        
                            indx_ = np.where( (spike <= t2_swap) & (spike > t1_swap))[0]
                            if indx_.shape[0] != 0:
                                spike_dict[k] = spike[indx_]
        except:
            print("rat 4? ")
        
        return spike_dict
    # def get_spikes(self, t1, t2, mod, selected_ids=None):
    #     selected_ids = selected_ids or self.rat[mod].keys()
    #     return {k: self.rat[mod][k][(self.rat[mod][k] > t1) & (self.rat[mod][k] <= t2)]
    #             for k in selected_ids if np.any((self.rat[mod][k] > t1) & (self.rat[mod][k] <= t2))}

    

    
    

    def log_L_i(self , time0 , k , spikes_dict , direction = 'past'):
        """Computes log likelihood for given neuron k at time t2."""
        spike = spikes_dict[k] # spike train (s)
        spike_timing_with_rep = [self.find_k(self.time, i) for i in spike] # time stamps corresponding to  (s)
        spike_timing, num_of_spikes = np.unique(self.time[spike_timing_with_rep], return_counts=True) # when spiking  , how many in the time bin 
        Rel_times_arr = time0 - spike_timing # spike timing reletive to t2 (s)

        if self.kernel_type == 'exp' or self.kernel_type == 'gap_exp':
            if direction == 'past':
                kernel = np.exp( -(Rel_times_arr) / self.Tau_indicies ) # use the Tau (s)
            elif direction == 'future':
                kernel = np.exp( (Rel_times_arr) / self.Tau_indicies ) # use the Tau (s)
        if self.kernel_type == 'const' or self.kernel_type == 'constgap' or self.kernel_type =='constsliding':
                kernel =np.abs( (Rel_times_arr) / self.Tau_indicies) 

        weighted_spikes = num_of_spikes * kernel
        sum_weighted_spikes = np.sum(weighted_spikes)
        
        single_tc  = self.all_tc[k]
        log_tc = np.log(single_tc)
        return sum_weighted_spikes * log_tc
    




    def get_error(self, t1 , t0 , Log_p , direction = 'past'):
        """Computes decoding error as Euclidean distance between true and predicted positions."""
        if direction == 'past':
            ind_ = range(int(t1), int(t0))
            true_x , true_y = self.x_traj[ind_][-1]+ self.Lim  , self.y_traj[ind_][-1] + self.Lim #converting to pixel 
        if direction == 'future':
            ind_ = range(int(t0), int(t1))
            true_x , true_y = self.x_traj[ind_][0]+ self.Lim  , self.y_traj[ind_][0] + self.Lim #converting to pixel 
        nonan_logp = np.nan_to_num(Log_p)
        # pred_x , pred_y = np.where(nonan_logp == np.max(nonan_logp)) # transpose and plus 1 (it gives index)
        pred_x , pred_y = np.where(nonan_logp == np.nanmax(Log_p)) # transpose and plus 1 (it gives index)
        pred_y += 1
        pred_x += 1
        pixel_dist = np.sqrt( (true_x - pred_x)**2 + (true_y - pred_y)**2)
        inst_speed = self.rat['speed'][ind_][-1]
        avg_speed = np.sqrt( (self.x_traj[ind_][-1] - self.x_traj[ind_][0])**2 + (self.y_traj[ind_][-1] - self.y_traj[ind_][0])**2 ) / (t0 - t1)
        return pixel_dist ,  inst_speed
        print('pred' , pred_y , pred_x , 'true' , true_y , true_x)
        true_x , true_y = x_traj[ind_][-1]  , y_traj[ind_][-1]
        true_radius = np.sqrt(true_x**2 + true_y**2)
        estimated_radius = np.sqrt((pred_x- Lim)**2 + (pred_y - Lim)**2 ) #converting to cm
        radius_dist = np.abs(true_radius - estimated_radius)
        return pixel_dist #, radius_dist
    


    def real_predicted_loc(self, t1 , t0 , Log_p , direction = 'past'):
        """Computes decoding error as Euclidean distance between true and predicted positions."""
        if direction == 'past':
            ind_ = range(int(t1), int(t0))
            true_x , true_y = self.x_traj[ind_][-1]+ self.Lim  , self.y_traj[ind_][-1] + self.Lim #converting to pixel 
        if direction == 'future':
            ind_ = range(int(t0), int(t1))
            true_x , true_y = self.x_traj[ind_][0]+ self.Lim  , self.y_traj[ind_][0] + self.Lim #converting to pixel 
        # if direction == 'present':


        nonan_logp = np.nan_to_num(Log_p)
        # pred_x , pred_y = np.where(nonan_logp == np.max(nonan_logp)) # transpose and plus 1 (it gives index)
        pred_x , pred_y = np.where(nonan_logp == np.nanmax(Log_p)) # transpose and plus 1 (it gives index)
        pred_y += 1
        pred_x += 1
        return (true_x, true_y) , (pred_x[0], pred_y[0])




    def get_confidence(self, t1, t0, Log_p, direction = 'past'):
        if direction == 'past':
            ind_ = range(int(t1), int(t0))
            true_x , true_y = self.x_traj[ind_][-1]+ self.Lim  , self.y_traj[ind_][-1] + self.Lim #converting to pixel 
        if direction == 'future':
            ind_ = range(int(t0), int(t1))
            true_x , true_y = self.x_traj[ind_][0]+ self.Lim  , self.y_traj[ind_][0] + self.Lim #converting to pixel 
        true_x = int(true_x)
        true_y = int(true_y)
        nonan_logp = np.nan_to_num(Log_p)

        pred_x , pred_y = np.where(nonan_logp == np.nanmax(Log_p)) # transpose and plus 1 (it gives index)
        pred_y += 1
        pred_x += 1

        confidence = nonan_logp[true_x][true_y]
        inst_speed = self.rat['speed'][ind_][-1]
        smoothed_logp =  gaussian_filter(Log_p, sigma=0.15, mode='constant', cval=0, truncate=3)  # mode and cval for consistency with matlab's conv2
        plt.imshow(smoothed_logp , aspect='auto' , origin='lower')
        plt.colorbar()
        plt.scatter(true_y, true_x, color='red', s=60, label='True position')
        plt.scatter(pred_y, pred_x, color='cyan', s=60, label='Predicted (MLE)')
        plt.title(f'{round(inst_speed, 2) , round(true_x , 4) , round(true_y , 4)}')
        plt.show()

        return confidence , inst_speed

    
    def get_confidence_MLE(self, t1, t0, Log_p, direction = 'past'):
        if direction == 'past':
            ind_ = range(int(t1), int(t0))
            # true_x , true_y = self.x_traj[ind_][-1]+ self.Lim  , self.y_traj[ind_][-1] + self.Lim #converting to pixel 
        if direction == 'future':
            ind_ = range(int(t0), int(t1))
            # true_x , true_y = self.x_traj[ind_][0]+ self.Lim  , self.y_traj[ind_][0] + self.Lim #converting to pixel 
        nonan_logp = np.nan_to_num(Log_p)
        # pred_x , pred_y = np.where(nonan_logp == np.max(nonan_logp)) # transpose and plus 1 (it gives index)
        # pred_x , pred_y = np.where(nonan_logp == np.nanmax(Log_p)) # transpose and plus 1 (it gives index)
        # pred_y += 1
        # pred_x += 1
        # print(pred_x, pred_y, nonan_logp.shape)
        # confidence = nonan_logp[pred_x[0]][pred_y[0]]
        confidence = np.max(nonan_logp)
        inst_speed = self.rat['speed'][ind_][-1]

        return confidence , inst_speed



    def decode_over_time_save_csv(self, time_indices = None , direction = 'past' , path = './decoding.csv'):
        """
        Run decoding over specified time points and return decoding errors
        as a csv file with breakdown of each neuron's contribution to the logL.
        csv format: t2 , error , speed , logL , logL_i
        Returns a key for neuron ID
        """
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_error = [[self.Tau , self.kernel_type , direction ]]
        neuron_ID = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 - self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 + self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + self.Tau_indicies


                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i

                error, speed = self.get_error(t1, t0, log_p , direction=direction)

                nonan_logp = np.nan_to_num(log_p)
                ind_ = np.where(nonan_logp == np.nanmax(log_p))
                #if direction == 'past':
                 #   ind_ = range(int(t1), int(t2))
                #if direction == 'future':
                 #   ind_ = range(int(t2), int(t1))
                #ind_true = self.x_traj[ind_][-1]+ self.Lim  , self.y_traj[ind_][-1] + self.Lim
                #print(ind_true , 'herehere')
                this_logL_ind_ = [n[ind_] for n in this_logL]

                if speed > self.speed_threshold:
                    final_info = [t0, int(error[0]), speed , np.nanmax(log_p)] + this_logL_ind_
                    all_error.append(final_info)
            
            neuron_ID = []
            for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        neuron_ID.append(k)

        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            with open(path , 'w' ,  newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_error)


            return all_error , neuron_ID
        
    def decode_over_time_fix_window(self, gap_mult, time_indices = None , direction = 'past' , path = './decoding.csv' ):
        """
        Run decoding over specified time points and return decoding errors
        as a csv file with breakdown of each neuron's contribution to the logL.
        csv format: t2 , error , speed , logL , logL_i
        Returns a key for neuron ID
        """
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                if self.cv>0:
                    time_indices = np.arange(self.Rel_times + self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                if self.cv > 0:
                    time_indices = np.arange(self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_error = [[self.Tau , self.kernel_type , direction ]]
        neuron_ID = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 -  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 - gap_mult * self.Tau_indicies
                        t1 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 +  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 + gap_mult * self.Tau_indicies
                        t1 = t2 +  self.Tau_indicies
                # print('t1: ',t1 ,'t2: ', t2 , 't0' , t0)

                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i

                error, speed = self.get_error(t1, t0, log_p , direction=direction)

                nonan_logp = np.nan_to_num(log_p)
                ind_ = np.where(nonan_logp == np.nanmax(log_p))

                this_logL_ind_ = [n[ind_] for n in this_logL]

                if speed > self.speed_threshold:
                    final_info = [t0, int(error[0]), speed , np.nanmax(log_p)] + this_logL_ind_
                    all_error.append(final_info)
            
            neuron_ID = []
            for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        neuron_ID.append(k)

        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            with open(path , 'w' ,  newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_error)




    def calculate_posterior(self, gap_mult, time_indices = None , direction = 'past' , path = './decoding.csv' ):
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                if self.cv>0:
                    time_indices = np.arange(self.Rel_times + self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                if self.cv > 0:
                    time_indices = np.arange(self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_posterior = []
        all_loc = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 -  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 - gap_mult * self.Tau_indicies
                        t1 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 +  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 + gap_mult * self.Tau_indicies
                        t1 = t2 +  self.Tau_indicies
                # print('t1: ',t1 ,'t2: ', t2 , 't0' , t0)

                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i
                
                all_posterior.append(log_p)
                all_loc.append(self.real_predicted_loc(t1 , t0 , log_p , direction = direction) ) # (true_x, true_y) , (pred_x, pred_y)


        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            return all_posterior , all_loc
            # with open(path , 'w' ,  newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(all_posterior)

    def decode_over_time_confidence(self, gap_mult, time_indices = None , direction = 'past' , path = './decoding.csv' ):
        
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                if self.cv>0:
                    time_indices = np.arange(self.Rel_times + self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                if self.cv > 0:
                    time_indices = np.arange(self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_error = [[self.Tau , self.kernel_type , direction ]]
        neuron_ID = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 -  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 - gap_mult * self.Tau_indicies
                        t1 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 +  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 + gap_mult * self.Tau_indicies
                        t1 = t2 +  self.Tau_indicies
                # print('t1: ',t1 ,'t2: ', t2 , 't0' , t0)

                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i

                error, speed = self.get_confidence_MLE(t1, t0, log_p , direction=direction)

                nonan_logp = np.nan_to_num(log_p)
                ind_ = np.where(nonan_logp == np.nanmax(log_p))

                this_logL_ind_ = [n[ind_] for n in this_logL]

                if speed > self.speed_threshold:
                    final_info = [t0, int(error), speed , np.nanmax(log_p)] + this_logL_ind_
                    all_error.append(final_info)
            
            neuron_ID = []
            for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        neuron_ID.append(k)

        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            with open(path , 'w' ,  newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_error)


            return all_error , neuron_ID
        
    def decode_over_time_corr(self, gap_mult, time_indices = None , direction = 'past' , path = './decoding.csv' ):
        
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                if self.cv>0:
                    time_indices = np.arange(self.Rel_times + self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                if self.cv > 0:
                    time_indices = np.arange(self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_error = [[self.Tau , self.kernel_type , direction ]]
        neuron_ID = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 -  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 - gap_mult * self.Tau_indicies
                        t1 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 +  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 + gap_mult * self.Tau_indicies
                        t1 = t2 +  self.Tau_indicies
                # print('t1: ',t1 ,'t2: ', t2 , 't0' , t0)

                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i

                error, speed = self.get_confidence_MLE(t1, t0, log_p , direction=direction)

                nonan_logp = np.nan_to_num(log_p)
                ind_ = np.where(nonan_logp == np.nanmax(log_p))

                this_logL_ind_ = [n[ind_] for n in this_logL]

                if speed > self.speed_threshold:
                    final_info = [t0, int(error), speed , np.nanmax(log_p)] + this_logL_ind_
                    all_error.append(final_info)
            
            neuron_ID = []
            for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        neuron_ID.append(k)

        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            with open(path , 'w' ,  newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_error)


            return all_error , neuron_ID
        
    
    def decode_over_time_segment(self, gap_mult, time_indices = None , direction = 'past' , path = './decoding.csv' ):
        """
        Run decoding over specified time points and return decoding errors
        as a csv file with breakdown of each neuron's contribution to the logL.
        csv format: t2 , error , speed , logL , logL_i
        Returns a key for neuron ID
        """
        modules = ['grid_mod1', 'grid_mod2', 'grid_mod3']
        if time_indices is None:
            if direction == 'past':
                if self.cv>0:
                    time_indices = np.arange(self.Rel_times + self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(self.Rel_times, self.num_points)
            if direction == 'future':
                if self.cv > 0:
                    time_indices = np.arange(self.ind_start_mask , self.ind_end_mask)
                else:
                    time_indices = np.arange(0 , self.num_points - self.Rel_times)
        all_error = [[self.Tau , self.kernel_type , direction ]]
        neuron_ID = []
       
        try:
            for t2 in time_indices:
                this_logL = []
                t0 = t2
                if direction == 'past':
                    t1 = t2 -  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2 = t2 -  gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 - gap_mult * self.Tau_indicies
                        t1 = t2 -  self.Tau_indicies
                if direction == 'future':
                    t1 = t2 +  self.Rel_times
                    if self.kernel_type == 'gap_exp' or self.kernel_type == 'constgap':
                        t2  = t2 + gap_mult * self.Tau_indicies
                    if self.kernel_type == 'constsliding':
                        t2 = t2 + gap_mult * self.Tau_indicies
                        t1 = t2 +  self.Tau_indicies
                # print('t1: ',t1 ,'t2: ', t2 , 't0' , t0)

                time1 = self.time[int(t1)]
                time2 = self.time[int(t2)]
                time0 = self.time[int(t0)]

                log_p = np.zeros((self.arena_shape_x, self.arena_shape_y))

                for mod in modules:
                    spikes_dict = self.get_spikes(time1, time2, mod)
                    for k in spikes_dict.keys():
                        log_l_i = self.log_L_i(time0, k, spikes_dict , direction = direction )
                        this_logL.append(log_l_i)
                        log_p += log_l_i

                true_loc , pred_loc = self.real_predicted_loc( t1 , t0 , log_p , direction = 'past') #(true_x, true_y) , (pred_x[0], pred_y[0])
                final_info = [t0, pred_loc , true_loc ] #+ this_logL_ind_
                all_error.append(final_info)
                # nonan_logp = np.nan_to_num(log_p)
                # ind_ = np.where(nonan_logp == np.nanmax(log_p))

                # this_logL_ind_ = [n[ind_] for n in this_logL]

                # if speed > self.speed_threshold:
                #     final_info = [t0, int(error[0]), speed , np.nanmax(log_p)] + this_logL_ind_
                #     all_error.append(final_info)
            
            # neuron_ID = []
            # for mod in modules:
            #         spikes_dict = self.get_spikes(time1, time2, mod)
            #         for k in spikes_dict.keys():
            #             neuron_ID.append(k)

        except KeyboardInterrupt:
            print(f'KeyboardInterrupt')
        except Exception as E:
            print("An error occurred:")
            traceback.print_exc()
        finally:
            with open(path , 'w' ,  newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_error)
            return final_info


