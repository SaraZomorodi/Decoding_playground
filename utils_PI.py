import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.ndimage import gaussian_filter
from utils import RAT


class TrajectorySegments():
    def __init__(self , edge_margin , arena_radius , RAT_ID  , LIGHTING):
        # arena parameters
        self.arena_radius = arena_radius   # cm
        self.edge_margin  = edge_margin    # cm
        self.valid_radius = arena_radius - edge_margin   

        rat = RAT(n=RAT_ID, lighting=LIGHTING, filter_speed=False) 
        self.t = rat.T  # (s), starts at 0 
        self.x = rat.X  # (cm) 
        self.y = rat.Y  # (cm) 
        self.v = rat.V  # (cm/s)
        
        self.segments = self.find_segments(self.find_valid_mask(), min_duration=0.5)
    

    def find_valid_mask(self):
        # distance from center
        r = np.sqrt(self.x**2 + self.y**2)

        # masks
        inside_mask = r <= self.valid_radius
        speed_mask = (self.v > 3) & (self.v < 100)

        valid_mask = inside_mask & speed_mask
        return valid_mask

    def find_segments(self, mask, min_duration=0.5):
        """
        Find contiguous True segments in mask.
        Returns list of (start_idx, end_idx, start_t, end_t, duration).
        end_idx is inclusive.
        """
        mask    = np.asarray(mask, dtype=bool)
        changes = np.diff(mask.astype(int))

        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0]

        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask) - 1]

        segments = []
        for s, e in zip(starts, ends):
            duration = self.t[e] - self.t[s]
            if duration >= min_duration:
                segments.append((s, e, self.t[s], self.t[e], duration))
        return segments


    def plot_traj(self, a , b ,segments ):
        theta = np.linspace(0, 2*np.pi, 400)

        plt.figure(figsize=(8, 8))
        plt.plot(self.x, self.y, color='lightgray', lw=1, label='full trajectory')

        # outer wall
        plt.plot(self.arena_radius * np.cos(theta), self.arena_radius * np.sin(theta), 'k--', label='arena wall')

        # valid interior boundary (5 cm away from edge)
        plt.plot(self.valid_radius * np.cos(theta), self.valid_radius * np.sin(theta), 'b--', label='valid region')

        for s, e, *_ in segments[a:b]:
            plt.plot(self.x[s:e+1], self.y[s:e+1], lw=2)

        plt.axis('equal')
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title(f"Segments with 3 < speed < 100 and at least {self.edge_margin} cm from edge")
        plt.legend()
        # plt.show()


    def using_segments(self, segments ):
        segment_data = []
        for s, e, ts, te, dur in segments:
            segment_data.append({
                "start_idx": s,
                "end_idx": e,
                "start_t": ts,
                "end_t": te,
                "duration": dur,
                "t": self.t[s:e+1],
                "x": self.x[s:e+1],
                "y": self.y[s:e+1],
                "v": self.v[s:e+1],
            })
        return segment_data




    def segment_stats(self):
        pass




class DecoderMLE():
    def __init__(self , RAT_ID , LIGHTING ):
        rat = RAT(n=RAT_ID, lighting=LIGHTING, filter_speed=False) 
        self.t = rat.T   # (s), starts at 0 
        self.x = rat.X   # (cm) 
        self.y = rat.Y   # (cm) 
        self.v = rat.V   # (cm/s)
        self.hd = rat.HD # rad
            
        # Choose which spikes to use 
        self.spikes_by_cell = {} 
        self.spikes_by_cell.update(rat.grid_mod1) 
        self.spikes_by_cell.update(rat.grid_mod2) 
        self.spikes_by_cell.update(rat.grid_mod3)

        x_edges, y_edges, x_centers, y_centers = self.make_position_bins()
        self.x_edges = x_edges 
        self.y_edges = y_edges
        self.x_centers = x_centers
        self.y_centers = y_centers


        self.rate_maps, self.occ = self.compute_rate_maps()



    # 1) Build spatial bins
    def make_position_bins(self, n_bins_x=25, n_bins_y=25, pad=1e-6):
        x_min, x_max = np.nanmin(self.x), np.nanmax(self.x)
        y_min, y_max = np.nanmin(self.y), np.nanmax(self.y)

        x_edges = np.linspace(x_min - pad, x_max + pad, n_bins_x + 1)
        y_edges = np.linspace(y_min - pad, y_max + pad, n_bins_y + 1)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        
        return x_edges, y_edges, x_centers, y_centers


    # 2) Occupancy map
    def compute_occupancy_map(self, speed=None, min_speed=None):
        """
        Occupancy in seconds for each spatial bin.
        Assumes x[t_k], y[t_k] represent sampled position at times t[k].
        """
        dt_samples = np.diff(self.t)
        if len(dt_samples) == 0:
            raise ValueError("t must contain more than one sample.")
        dt0 = np.median(dt_samples)

        # One dt per position sample
        dt_per_sample = np.empty_like(self.t, dtype=float)
        dt_per_sample[:-1] = dt_samples
        dt_per_sample[-1] = dt_samples[-1]

        valid = np.isfinite(self.x) & np.isfinite(self.y) & np.isfinite(dt_per_sample)

        if (speed is not None) and (min_speed is not None):
            valid &= (speed >= min_speed)

        occ, _, _ = np.histogram2d(
            self.x[valid], self.y[valid],
            bins=[self.x_edges, self.y_edges],
            weights=dt_per_sample[valid])

        return occ



    def compute_spike_map_for_cell(self, spike_times,
                                speed=None, min_speed=3.0):
        """
        Assign each spike to the nearest sampled position index using searchsorted.
        Filters spikes based on the rat's speed at the time of firing.
        """
        spike_times = np.asarray(spike_times, dtype=float)
        
        # 1. Keep spikes only within the tracking time range
        valid_time_mask = (spike_times >= self.t[0]) & (spike_times <= self.t[-1])
        spike_times = spike_times[valid_time_mask]

        if len(spike_times) == 0:
            return np.zeros((len(self.x_edges)-1, len(self.y_edges)-1), dtype=float)

        # 2. Use searchsorted to find the index in 't' that is closest to each spike_time
        # This finds where spike_times would be inserted to maintain order
        indices = np.searchsorted(self.t, spike_times)
        
        # Prevent index out of bounds for spikes exactly at t[-1]
        indices[indices == len(self.t)] = len(self.t) - 1
        
        # 3. Extract position and speed at those specific indices
        spike_x = self.x[indices]
        spike_y = self.y[indices]
        
        # 4. Apply Speed Filter
        valid_mask = np.isfinite(spike_x) & np.isfinite(spike_y)
        
        if speed is not None:
            spike_speed = speed[indices]
            # Filter: speed must be >= 3 cm/s
            valid_mask &= (spike_speed >= min_speed)

        # 5. Create the 2D histogram (Spike Map)
        spike_map, _, _ = np.histogram2d(
            spike_x[valid_mask], spike_y[valid_mask],
            bins=[self.x_edges, self.y_edges]
        )

        return spike_map


    # 4) Rate maps for all cells
    def compute_rate_maps(self,
                        speed=None, min_speed=None,
                        smooth_sigma=1.0, occ_epsilon=1e-6):
        """
        Returns:
            rate_maps: dict[cell_id] -> 2D firing rate map (Hz)
            occ: 2D occupancy map (s)
        """
        occ = self.compute_occupancy_map(
            speed=speed, min_speed=min_speed
        )

        if smooth_sigma is not None and smooth_sigma > 0:
            occ_smooth =gaussian_filter(occ, smooth_sigma, mode="constant")
        else:
            occ_smooth = occ.copy()

        rate_maps = {}

        for cell_id, spike_times in self.spikes_by_cell.items():
            spike_map = self.compute_spike_map_for_cell(
                spike_times,
                speed=speed, min_speed=min_speed
            )

            if smooth_sigma is not None and smooth_sigma > 0:
                spike_map = gaussian_filter(spike_map, smooth_sigma, mode="constant")

            rate_map = spike_map / (occ_smooth + occ_epsilon)
            rate_maps[cell_id] = rate_map

        return rate_maps, occ


    # 5) Count spikes in decoding windows
    def count_spikes_in_windows(self, spike_times, window_edges):
        """
        Returns counts for consecutive windows defined by window_edges.
        """
        spike_times = np.asarray(spike_times, dtype=float)
        counts, _ = np.histogram(spike_times, bins=window_edges)
        return counts


    # 6) Decode one segment
    def decode_segment_bayes_uniform(self , segment,
                                     dt_decode=0.1,
                                    rate_floor=1e-12):
        """
        Bayesian decoding for one segment [s:e] with uniform prior.

        segment: [s, e, t[s], t[e], duration]
        """
        s, e = int(segment[0]), int(segment[1])

        t_start = self.t[s]
        t_end   = self.t[e]

        if t_end <= t_start:
            raise ValueError("Segment end time must be greater than start time.")

        # Window edges for decoding
        window_edges = np.arange(t_start, t_end + dt_decode, dt_decode)
        if window_edges[-1] < t_end:
            window_edges = np.append(window_edges, t_end)

        n_windows = len(window_edges) - 1
        cell_ids = list(self.rate_maps.keys())
        n_cells = len(cell_ids)

        # Spike count matrix: [n_windows, n_cells]
        K = np.zeros((n_windows, n_cells), dtype=int)
        for ci, cell_id in enumerate(cell_ids):
            K[:, ci] = self.count_spikes_in_windows(self.spikes_by_cell[cell_id], window_edges)

        # Stack rate maps: [n_cells, nx, ny]
        lam = np.stack([self.rate_maps[cell_id] for cell_id in cell_ids], axis=0)
        lam = np.maximum(lam, rate_floor)  # avoid log(0)

        nx, ny = lam.shape[1], lam.shape[2]

        # Precompute for speed
        log_lam_dt = np.log(lam * dt_decode)         # [n_cells, nx, ny]
        lam_dt = lam * dt_decode                     # [n_cells, nx, ny]

        decoded = []
        posteriors = []

        for w in range(n_windows):
            k = K[w]  # [n_cells]

            # log posterior up to constant:
            # sum_i [k_i log(lam_i*dt) - lam_i*dt]
            log_post = np.sum(
                k[:, None, None] * log_lam_dt - lam_dt,
                axis=0
            )  # [nx, ny]

            # uniform prior -> nothing added

            # stabilize before exponentiating
            log_post = log_post - np.max(log_post)
            post = np.exp(log_post)
            post /= np.sum(post)

            # MAP estimate
            ix, iy = np.unravel_index(np.argmax(post), post.shape)
            x_hat = 0.5 * (self.x_edges[ix] + self.x_edges[ix+1])
            y_hat = 0.5 * (self.y_edges[iy] + self.y_edges[iy+1])

            # true position = midpoint of the time window
            t_mid = 0.5 * (window_edges[w] + window_edges[w+1])
            x_true = np.interp(t_mid, self.t, self.x)
            y_true = np.interp(t_mid, self.t, self.y)

            decoded.append([t_mid, x_hat, y_hat, x_true, y_true])
            posteriors.append(post)

        decoded = np.asarray(decoded)           # [n_windows, 5]
        posteriors = np.asarray(posteriors)     # [n_windows, nx, ny]

        return decoded, posteriors, K, window_edges


    # 7) Error metric
    def decoding_error(self, decoded):
        dx = decoded[:, 1] - decoded[:, 3]
        dy = decoded[:, 2] - decoded[:, 4]
        err = np.sqrt(dx**2 + dy**2)
        return err



    def plot_colored_trajectory(self, x, y, c, ax=None, cmap='viridis', linewidth=2, label='',
                                vmin=None, vmax=None, alpha=1.0, zorder=3):
        x = np.asarray(x)
        y = np.asarray(y)
        c = np.asarray(c)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        nseg = len(segments)
        c = c[:nseg]

        lc = LineCollection(segments, cmap=cmap, linewidth=linewidth,
                            alpha=alpha, zorder=zorder)
        lc.set_array(c)

        if vmin is not None or vmax is not None:
            lc.set_clim(vmin=vmin, vmax=vmax)

        ax.add_collection(lc)
        ax.autoscale()   # <- important

        return lc, ax


    def segment_velocity(self , decoded ,segment ,dt_decode = 0.1 ,  mod = 'inter'):
        if mod == 'inter':
            decoded_t = decoded[:, 0]
            decoded_speed = np.interp(decoded_t, self.t, self.v)
        if mod == 'last_point':
            
            s, e = int(segment[0]), int(segment[1])
            dt_ind =  (s - e) / len(decoded[:, 0])
            window_edges = np.arange(s, e , dt_ind)
            decoded_speed = np.array([self.v[i] for i in window_edges])
        return decoded_speed


    def plot_comparitive_plot(self , decoded):
        fig, ax = plt.subplots(figsize=(8, 8))

        # speed at decoded times
        decoded_t = decoded[:, 0]
        decoded_speed = np.interp(decoded_t, self.t, self.v)

        vmin = np.nanmin(decoded_speed)
        vmax = np.nanmax(decoded_speed)

        # true trajectory as color-coded line
        lc1, ax = self.plot_colored_trajectory(
            decoded[:, 3], decoded[:, 4], decoded_speed,
            ax=ax, cmap='Blues', linewidth=3,
            vmin=vmin, vmax=vmax
        )

        # predicted locations as dots
        ax.scatter(
            decoded[:, 1], decoded[:, 2],
            s=18, c='red', alpha=0.9, zorder=4, label='Predicted location'
        )

        # thin lines connecting true and predicted points
        for i in range(len(decoded)):
            ax.plot(
                [decoded[i, 3], decoded[i, 1]],
                [decoded[i, 4], decoded[i, 2]],
                color='gray', linewidth=0.6, alpha=0.6, zorder=2
            )

        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title('True trajectory colored by speed with decoded positions')
        ax.set_aspect('equal')
        ax.autoscale()

        cbar = plt.colorbar(lc1, ax=ax)
        cbar.set_label('Speed (cm/s)')

        ax.legend()
        # plt.show()


    def get_distance_based_windows(self , t, x, y, dist_step=5.0):
        """
        Creates window edges such that the rat travels roughly 'dist_step' 
        centimeters in each window.
        """
        # Calculate cumulative distance
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dr = np.sqrt(dx**2 + dy**2)
        cum_dist = np.cumsum(np.nan_to_num(dr))
        
        # Determine the distance values at which to create a boundary
        total_dist = cum_dist[-1]
        distance_checkpoints = np.arange(0, total_dist, dist_step)
        
        # Map these distances back to time using interpolation
        # (Since cum_dist is monotonically increasing, we can flip it)
        window_edges = np.interp(distance_checkpoints, cum_dist, t)
        
        return window_edges
    

    def decode_segment_adaptive(self , segment, dist_step=5.0, rate_floor=1e-12):
        s, e = int(segment[0]), int(segment[1])
        
        # Get adaptive window edges for this specific segment's trajectory
        window_edges = self.get_distance_based_windows(self.t[s:e], self.x[s:e], self.y[s:e], dist_step)
        
        n_windows = len(window_edges) - 1
        cell_ids = list(self.rate_maps.keys())
        
        # Pre-stack rate maps [n_cells, nx, ny]
        lam = np.stack([self.rate_maps[cid] for cid in cell_ids], axis=0)
        lam = np.maximum(lam, rate_floor)
        
        decoded = []
        
        for w in range(n_windows):
            t_start, t_end = window_edges[w], window_edges[w+1]
            dt = t_end - t_start
            if dt <= 0: continue
                
            # Count spikes in this specific adaptive window
            counts = []
            for cid in cell_ids:
                spks = self.spikes_by_cell[cid]
                counts.append(np.sum((spks >= t_start) & (spks < t_end)))
            k = np.array(counts)
            
            # Bayesian Log-Likelihood with window-specific dt
            # log P ~ sum( k*log(lam*dt) - lam*dt )
            log_post = np.sum(k[:, None, None] * np.log(lam * dt) - (lam * dt), axis=0)
            
            # Maximize stability and normalize
            log_post -= np.max(log_post)
            post = np.exp(log_post)
            post /= np.sum(post)
            
            # MAP Estimate
            ix, iy = np.unravel_index(np.argmax(post), post.shape)
            x_hat = 0.5 * (self.x_edges[ix] + self.x_edges[ix+1])
            y_hat = 0.5 * (self.y_edges[iy] + self.y_edges[iy+1])
            
            # Ground truth at midpoint
            t_mid = (t_start + t_end) / 2
            x_true = np.interp(t_mid, self.t, self.x)
            y_true = np.interp(t_mid, self.t, self.y)
            
            decoded.append([t_mid, x_hat, y_hat, x_true, y_true, dt])

        return np.array(decoded)
    

    



class MakePlots():
    def __init__(self , edge_margin= 15 , arena_radius= 75 , RAT_ID=2 , LIGHTING='light' ):
        self.edge_margin    = edge_margin
        self.arena_radius   = arena_radius
        self.RAT_ID         = RAT_ID
        self.LIGHTING       = LIGHTING


        self.rat = TrajectorySegments(edge_margin= self.edge_margin , arena_radius= self.arena_radius , RAT_ID=self.RAT_ID , LIGHTING=self.LIGHTING)
        self.segments = self.rat.segments
        self.rat_decode = DecoderMLE(RAT_ID=self.RAT_ID , LIGHTING=self.LIGHTING)

    def get_error_vs_distance_traveled(self, dist_step=8.0):
        all_dists = []
        all_errors = []

        for seg in self.segments:
            s, e = int(seg[0]), int(seg[1])
            
            # 1. Decode the segment
            dec = self.rat_decode.decode_segment_adaptive(seg , dist_step = dist_step)
            if len(dec) == 0: continue

            # 2. Calculate cumulative distance for the WHOLE segment trajectory
            seg_x, seg_y = self.rat_decode.x[s:e], self.rat_decode.y[s:e]
            dx = np.diff(seg_x, prepend=seg_x[0])
            dy = np.diff(seg_y, prepend=seg_y[0])
            dr = np.sqrt(dx**2 + dy**2)
            cum_dist_full = np.cumsum(np.nan_to_num(dr))
            t_seg = self.rat_decode.t[s:e]

            # 3. Interpolate distance at the decoding midpoints (dec[:, 0])
            dist_at_windows = np.interp(dec[:, 0], t_seg, cum_dist_full)
            
            # 4. Error calculation
            err = np.sqrt((dec[:, 1] - dec[:, 3])**2 + (dec[:, 2] - dec[:, 4])**2)
            
            all_dists.extend(dist_at_windows)
            all_errors.extend(err)
            
        return np.array(all_dists), np.array(all_errors)



    def plot_traveled_space(self,  dist_step=8.0 , dist_bins_ = 5.0):
        dist_vals, err_vals = self.get_error_vs_distance_traveled(dist_step=dist_step)

        # 1. Binning by distance (e.g., every 5 cm)
        d_bins = np.arange(0, np.percentile(dist_vals, 95), dist_bins_) 
        bin_centers = 0.5 * (d_bins[:-1] + d_bins[1:])
        
        bin_means, bin_sem, bin_counts = [], [], []

        for i in range(len(d_bins)-1):
            mask = (dist_vals >= d_bins[i]) & (dist_vals < d_bins[i+1])
            count = np.sum(mask)
            bin_counts.append(count)
            
            if count > 0:
                bin_means.append(np.mean(err_vals[mask]))
                bin_sem.append(np.std(err_vals[mask]) / np.sqrt(count))
            else:
                bin_means.append(np.nan)
                bin_sem.append(np.nan)

        # 2. Create the figure and the first axis (Error)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Error Line (Primary Y-axis)
        ax1.errorbar(bin_centers, bin_means, yerr=bin_sem, fmt='-s', 
                    color='darkorange', ecolor='gray', capsize=3, lw=2, label='Mean Error')
        ax1.set_xlabel('Cumulative Distance Traveled in Segment (cm)', fontsize=12)
        ax1.set_ylabel('Mean Decoding Error (cm)', color='darkorange', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='darkorange')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 3. Create the second axis (Sample Count)
        ax2 = ax1.twinx()
        # Plot Sample Count Bars (Secondary Y-axis)
        # width matches the bin size (5.0) with a slight gap for aesthetics
        ax2.bar(bin_centers, bin_counts, width=4.0, alpha=0.15, color='gray', label='Sample Count')
        ax2.set_ylabel('Number of Windows (Sample Count)', color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')

        plt.title(f'Decoding Performance & Data Density vs. Path Distance (step={dist_step}cm)', fontsize=14)
        fig.tight_layout()
        # plt.show()

