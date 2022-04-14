import stumpy
import time
import numpy as np
from numba import cuda
from typing import Tuple
from scipy.signal import decimate
from readers import Reader
from utils import consecutive


class MotifFinder:
    def __init__(self, rd: Reader, decimate_q=15, lowpass=150, filter_order=3):
        self.rd = rd
        self.decimate_q = decimate_q
        self.lowpass = lowpass
        self.filter_order = filter_order
        self.motives = []
        self.all_gpu_devices = [device.id for device in cuda.list_devices()]  # Get a list of all available GPU devices

    def search(self, t_start, t_stop, durations=(2, 0.4), is_remove_motives=True):
        v, t = self.read_signal(t_start, t_stop)
        for duration in durations:
            motif_indices, window = self.find_duration_motif(v, duration)
            good_motives_idx = self.analyze_motives(motif_indices, window)
            for i in good_motives_idx:
                self.motives.append(v[motif_indices[i, 0]:motif_indices[i, 0]+window])
            if is_remove_motives:
                v, t = self.remove_motives_from_signal(v, t, window, motif_indices)

    def find_duration_motif(self, v, window_duration, max_motifs=8) -> Tuple[np.ndarray, int]:
        window = int(self.rd.fs * window_duration / (self.decimate_q or 1))
        t0 = time.time()
        mps = stumpy.gpu_stump(v, m=window, device_id=self.all_gpu_devices)
        print(f'Finish calculating matrix profile with window of {window_duration} seconds in {(time.time() - t0) / 60:.1f} minutes.')
        motif_distances, motif_indices = stumpy.motifs(v, mps[:, 0], max_motifs=max_motifs, max_matches=1000)
        return motif_indices, window

    @staticmethod
    def analyze_motives(motif_indices, window, threshold=0.8):
        # calculate probability of overlapping motives
        M = np.zeros((motif_indices.shape[0], motif_indices.shape[0]))
        for row1 in range(motif_indices.shape[0]):
            for row2 in range(motif_indices.shape[0]):  # np.arange(mi.shape[0]-1, row1, -1):
                for i in motif_indices[row1, :]:
                    if i == -1:
                        break
                    if any(i <= k <= (i + window) for k in motif_indices[row2, :]) or \
                            any(i <= k + window <= (i + window) for k in motif_indices[row2, :]):
                        M[row1, row2] += 1
            l = np.where(motif_indices[row1, :] == -1)[0]
            M[row1, :] /= l[0] if len(l) > 0 else motif_indices.shape[1]

        # Eliminate overlapping motives
        motives = list(range(M.shape[0]))
        for i in motives.copy():
            if i not in motives:
                continue
            for j in [m for m in motives if m != i]:
                if M[i, j] >= 0.8:
                    motives.remove(j)
        return motives

    def plot_motif_rate_vs_time(self, motif, threshold, t_start=None, t_stop=None):
        v, t = self.read_signal(t_start, t_stop)
        t0 = time.time()
        dists = stumpy.mass(motif, v)
        print(f'Finish motif search in {(time.time()-t0)/60:.1f} minutes.')
        idx = []
        for idx_group in consecutive(np.where(dists < threshold)[0]):
            if len(idx_group) == 1:
                idx.append(idx_group[0])
            else:
                idx.append(np.argmin(dists[idx_group]) + idx_group[0])

    def read_signal(self, t_start, t_stop):
        i_start = int(t_start * self.rd.fs) if t_start else None
        i_stop = int(t_stop * self.rd.fs) if t_stop else None
        v, t = self.rd.read(i_start=i_start, i_stop=i_stop, lowpass=self.lowpass, filter_order=self.filter_order)
        if self.decimate_q:
            v, t = decimate(v, self.decimate_q), decimate(t, self.decimate_q)
        return v, t

    @staticmethod
    def remove_motives_from_signal(v, t, window, motif_indices):
        idx = np.full(v.shape, True)
        for i in motif_indices.flatten():
            idx[i:i + window] = False
        return v[idx], t[idx]
