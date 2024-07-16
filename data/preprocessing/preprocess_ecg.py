import os
import numpy as np
from scipy.signal import resample
from matplotlib.mlab import psd
from scipy import signal
from ecgdetectors import Detectors

def preprocess_ecg(lead_signal):

    mask = np.isnan(lead_signal)
    lead_signal[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), lead_signal[~mask])
    diff = lead_signal[20:]-lead_signal[0:-20]
    perc_zeros = np.sum(np.absolute(diff) < 1e-08)/len(lead_signal)
    if perc_zeros > .20:
        return

    lead_signal = resample(lead_signal, num=len(lead_signal)*100//125, axis=0)

    power, psd_frequencies = psd(lead_signal[:,0], NFFT=1024, Fs=100,detrend="linear")

    cutofffreq = 20
    power = power[psd_frequencies < cutofffreq]/np.max(power[psd_frequencies <cutofffreq]) 

    peaks1, _ = signal.find_peaks(power,distance=9,prominence=.03)
    if len(peaks1) > 0 and np.max(peaks1) > np.sum(psd_frequencies < 10):
        std_diffbetweenpeaks = np.std(peaks1[1:] - peaks1[:-1])
        if std_diffbetweenpeaks < 1:
            return
        else:
            peaks2, _ = signal.find_peaks(power,distance=3,prominence=.03)
            new_peaks = set(peaks1).symmetric_difference(set(peaks2))
            closest = [peaks1[np.abs(peaks1 - x).argmin()] for x in new_peaks]
            if len(new_peaks) != 0:
                avgdiff_afterrelaxingdistance = np.average(np.abs(power[list(new_peaks)] - power[closest]))
            else:
                avgdiff_afterrelaxingdistance = 0

            if std_diffbetweenpeaks < 5 and \
               (avgdiff_afterrelaxingdistance > .25 or len(new_peaks) <= len(peaks1)//2): # allow for atrial fibrlation frequency changes to pass through
                pass
            else:
                return
    else:
        return

    detectors = Detectors(100)
    rangeofpossiblestarts = len(lead_signal) - 100*60*5
    if rangeofpossiblestarts == 0:
        startidx = 0
    elif rangeofpossiblestarts < 0:
        return
    else:
        startidx = np.random.randint(rangeofpossiblestarts)

    lead_signal = lead_signal[startidx:startidx+100*60*5, 0].astype(np.float32) # shape = [1000, 1 channel]
    r_peaks = detectors.pan_tompkins_detector(lead_signal)
    if len(r_peaks) <= 125:
        return

    def find_mode(hist, bin_edges):
        max_idx = np.argwhere(hist == np.max(hist))[0][0]
        mode = np.mean([bin_edges[max_idx], bin_edges[1+max_idx]])
        
        return mode

    # mode center and bound values
    hist, bin_edges = np.histogram(lead_signal, bins=50) # hist shape [50, ]
    mode = find_mode(hist, bin_edges)
    lead_signal -= mode
    max_val = np.max(np.abs(lead_signal))
    lead_signal /= max_val/1


    return lead_signal

    




    

