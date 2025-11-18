# Libraries
import sys
import pywt
import time
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
import pycwt as wavelet
import os
import json
import datetime
import seaborn as sns
import sklearn as sk
from sklearn.decomposition import PCA, FastICA

from joblib import Parallel, delayed
import multiprocessing
from itertools import product

# Import listed colormap
from matplotlib.colors import ListedColormap
import matplotlib.dates as md

# Add my module to python path
sys.path.append("../../")

# Own libraries
#from datasets.load import load_setup
#from datasets.load import load_matfiles
#from pyscripts.visualization.graphics.plots import *

# Scipy
from scipy import signal, ndimage, stats




warnings.simplefilter("ignore")


# -------------------------------------------------------------------
#                            helpers
# -------------------------------------------------------------------
def fnorm(v, fs=939, method='nyquist'):
    """This method normalizes a frequency vector

    Parameters
    ----------
    v : np.array
      The vector of frequencies

    fs : float-like
      The sample frequency

    method : string
      The normalization method

    Returns
    -------
      np.array
    """
    # Normalize
    if 'nyquist':
        return np.array(v) / (fs * 0.5)
    # Return
    return None

def computeFastICA(X, n_components=None, random_state=0):
    # compute ICA:  X = AxS   A: mixing matrix   S: source matrix
    icaTransformer = FastICA(n_components=n_components, random_state=random_state)   # random_state pass an int, for reproducible results across multiple function calls. 
    S_ = icaTransformer.fit_transform(X)  # Get the estimated sources
    A_ = icaTransformer.mixing_  # Get estimated mixing matrix
    return (S_, A_)

def convertDfType(df, typeFloat='float16'):
    for col in df.columns:
        if col.startswith('ch_'):
            df[col] = df[col].astype(typeFloat)
    return df

def peaks(v, **kwargs):
    """This method finds the peaks of a signal

    Parameters
    ----------
    signal : np.array
      The signal vector

    Returns
    -------
    peaks: np.array
      The vector indicating where peaks were found

    peaks_idxs: array
      The vector with the peaks indexes

    peaks_options: array
      The vector with additional information of the peaks
    """
    # Compute peaks
    peaks_idxs, peaks_options = signal.find_peaks(v, **kwargs)

    # Create peaks vector
    peaks = np.zeros(v.shape)
    peaks[peaks_idxs] = 1

    # Return
    return peaks, peaks_idxs, peaks_options


def downsampling(data, Nd, fs, dt):
    """
    Downsample the the data
        some operations should not be performed at the full sampling
        rate and repeatedly downsampling the data is inefficient
    """

    print('Downsampling signal')
    signal2 = signal.resample(data, Nd)

    return signal2, fs / Nd, dt * Nd


def get_spikes_online(data, spike_window=80, tf=5, offset=10, max_thresh=350):
    # Doesn't work well: doesn't filter out peaks over max_thresh
    data = data.values

    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) * tf
    print(thresh)

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0]
    pos = pos[pos > spike_window]
    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window * 2])
    for i in pos:
        if i < data.shape[0] - (spike_window + 1):

            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i: i + spike_window * 2]

            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) + i

                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

                # Append data
                spike_samp = np.append(spike_samp, tmp_samp)
                wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window*2), axis=0)

    # Remove duplicates
    ind = np.where(np.diff(spike_samp) > 1)[0]
    spike_samp = spike_samp[ind]
    wave_form = wave_form[ind]
    # print(ind)
    print(len(spike_samp))
    # sys.exit()

    return spike_samp, wave_form

def legend_without_duplicate_labels(ax, bbox_to_anchor):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(handles, labels, loc='lower right', bbox_to_anchor=bbox_to_anchor)
    return ax.legend(*zip(*unique))

def get_spikesAG(data, fs, spike_window=80, tf=5, offset=10, max_thresh=350):
    data = data.values

    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) * tf
    print(thresh)

    # Find positions wherere the threshold is crossed
    pos, peaks_options = signal.find_peaks(data, height=[thresh, max_thresh],
                                                 distance=0.001 * fs)
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    wave_form = np.zeros([0, spike_window * 2 + 1])

    for i in pos:
        # Re-center window on maximum sample
        tmp_waveform = data[i - spike_window:i + spike_window + 1]

        # Append data
        wave_form = np.vstack((wave_form, tmp_waveform))

    return pos, wave_form



def get_spikes(data, fs, cardiac, spike_window,min_thr,half_width, C, Cmax, find_peaks_args, window=None, neo=False, verbose=False):
    """
    - This method detect peaks (maxima)
    The parameter 'distance' for find-peaks guarantees the minimum separation
    between 2 peaks so that windows are not overlapped
    - Returns also duplicates

    Parameters
    ---------

    Returns
    -------
    """
    if verbose:
        print('Entered in get_spikes()...')
        print('find_peaks_args: %s' %print(find_peaks_args))
    if neo:
        # Apply NEO filter to signal
        print(data)
        data = NEO(data)
    
    if 'height' not in find_peaks_args:
        find_peaks_args['height'] = (C * np.mean(np.abs(data)/0.6745), Cmax * np.mean(np.abs(data)/0.6745))
        if verbose:
            print('Mean of noise cardiac: %s' %np.mean(np.abs(data)/0.6745))
        # AG on 18/11 uncomment below if problems with cardiac threshold
        #find_peaks_args['height'] = (C * np.mean(np.abs(data)), ) #np.abs(data).max()-20)
        #print('height not in find_peaks_args')

    # Find positions where the threshold is crossed
    if verbose:
        print(find_peaks_args)
    pos, peaks_options = signal.find_peaks(data, **find_peaks_args)

    #pos = pos[pos > spike_window[0]]
    #pos = pos[pos < (len(data) - spike_window[1])]

    #wdws = [range(i - spike_window[0], i + spike_window[1] + 1) for i in pos]
    #wdws = np.stack(np.array(wdws))
    if verbose:
        print('Peaks detected.')
    """
    # Plot to check
    fig, ax = plt.subplots(1,1, sharex=True)
    ax.plot(data.values)
    ax.plot(pos, data.values[pos], 'x')
    plt.show()
    sys.exit()
    """
    return pos #, data[wdws]

def get_spikes_simpleTh(data, fs, spike_window, C, find_peaks_args, neo=False):
    """
    This method does similar to get_spikes, but instead of identifying the maxima of the peaks
    it detects where the threshold is crossed 
    Detects all peaks inclding cardiac
    Also returns duplicates as old code was removed
    Parameters
    ---------

    Returns
    -------
    """
    print('Detecting Index first edge...')
    if neo:
        # Apply NEO filter to signal
        data = NEO(data)
        #find_peaks_args['height'] = (C * np.mean(np.abs(data)),)
        find_peaks_args['height'] = (np.median(np.abs(data)/0.6745), )

    # Find positions wherere the threshold is crossed
    #pos_data = np.where(data > find_peaks_args['height'])[0]
    pos_data = (data > find_peaks_args['height']).astype(int)
    # print(sum(pos_data==1))
    trig = np.diff(pos_data)   # 1st (+) and 2nd (-) edge of crossings
    first_edge = (trig>0).astype(int)         # Equals 1 only at beginning of peak and 0 otherwise
    #print(np.where(first_edge==1))
    index_first_edge = np.where(first_edge==1)[0]+1
    #print(sum(index_first_edge<0))

    """
    # Plot to check
    fig, ax = plt.subplots(1,1, sharex=True)
    ax.plot(data.values)
    ax.plot(index_first_edge, data.values[index_first_edge], 'x')
    ax.plot(np.arange(len(data)), np.ones(len(data))*find_peaks_args['height'])
    plt.show()
    sys.exit()
    """
    """ CODE BELOW MOVED TO max_centered_peaks 
    index_first_edge = index_first_edge[index_first_edge > spike_window[0]]
    index_first_edge = index_first_edge[index_first_edge < (len(data) - spike_window[1])]
    #print(index_first_edge)

    waveforms = [] #np.zeros(len(index_first_edge), 2 * spike_window[0] + 1);
    pos = [] #np.zeros(len(index_first_edge), 1);
    s=0
    for i in index_first_edge:
        
        index_range = np.arange(i - spike_window[0], i + spike_window[0] )  # waveforms is returned and only used for size
        #print(index_range)
        sig_local = data[index_range]
        peak_index = np.argmax(sig_local)                   # Local index within that array
        new_peak_pos = i - spike_window[0] + (peak_index)   # General index in data
        #print(i - spike_window[0])
        #print(peak_index)
        #print(new_peak_pos)
        #if i == index_first_edge[3]:   
            #sys.exit()
        
        if new_peak_pos in cardiac_peaks:
            #print(s)
            s=s+1
        #if new_peak_pos not in pos and \
        #    new_peak_pos < (len(data) - spike_window[1]) and \
        #    new_peak_pos not in cardiac_peaks:
        if new_peak_pos not in cardiac_peaks:
            if new_peak_pos not in pos:
                pos.append(new_peak_pos)
                #print(index_range[0] + peak_index - 1 - spike_window[1])
                #print(index_range[-1] + peak_index - 1 - spike_window[1])
                new_window = np.arange(new_peak_pos - spike_window[0], new_peak_pos + spike_window[1] )

                #print(new_window)
                #print(new_peak_pos)
                waveforms.append(data[new_window])
                #sys.exit()
    print(s)    
    #sys.exit()
    wdws = np.stack(np.array(waveforms))
    return pos, wdws
    """
    print('Index first edge detected.')
    
    return index_first_edge

def combine_threshold_crossing(indspos, indsneg):
    #The following function does similar to combine_threshold_crossing in Zanos 
    # (although different implementation)
    '''
     Wilson: 'combine_threshold_crossing' takes indspos and indsneg as input, but output an array of indices
              that is not quite the same as the union of the positive and negative peaks. (see report for reference)
     It seems that although 'combine_threshold_crossing' would be able to give a rough indication of
    where the spikes (boundary of image) are, the output indices do not accurately reflet the exact
    position of the spikes and the corresponding amplitude. 

    For these reasons, the function ‘combine_threshold()’ was created and used instead.
    '''
 
    ind_combined = np.union1d(indspos, indsneg)
    indsvec = np.zeros(np.max(ind_combined)+1, dtype=np.int)
    indsvec[ind_combined] = 1
    closed = ndimage.binary_dilation(indsvec, iterations=20).astype(np.int)
    spikes_idx = np.where(np.diff(closed)==1)[0]
    return np.array(spikes_idx)

# The new function below is created by Wilson
def combine_threshold(indspos, indsneg, fs, min_time_separation = 0.001):
    # verbosing the input (probably unneccesary)--Wilson
    print("\n" * 3)
    print(f'len of indspos: {len(indspos)}')
    # print(f'indspos: {(indspos)}')
    print(f'len of indsneg: {len(indsneg)}')
    # print(f'indsneg: {(indsneg)}')
    print("\n" * 3)

    # calculate the minimal index seperation of peaks for them to be counted as distinct neural spikes
    min_idx_separation = int(fs*min_time_separation)
    print(f'the min_idx_separation is: {min_idx_separation}')

    #combine the input and delete spikes that are too close together
    ind_combined = np.union1d(indspos, indsneg)
    print(f'the num of index_first_edge initially after combination: {len(ind_combined)}')
    spikes_idx = np.delete(ind_combined, np.argwhere(np.ediff1d(ind_combined) <= min_idx_separation) + 1)
    index_eliminated = ind_combined[np.argwhere(np.ediff1d(ind_combined) <= min_idx_separation) + 1]
    # print(f'index_eliminated:{index_eliminated}')
    
    #verbosing 
    print(f'the num of index_first_edge after elimination: {len(np.array(spikes_idx))}')
    num_peaks_eliminated = len(ind_combined)-len(np.array(spikes_idx))
    print(f'the num of index_first_edge eliminated: {num_peaks_eliminated}')
    # print(f'the output index:{np.array(spikes_idx)}')
    
    return np.array(spikes_idx)

def max_centered_peaks(data,index_first_edge, spike_window, cardiac_peaks=[]):
    print("Aligning peaks to max and getting waveform")
    index_first_edge = index_first_edge[index_first_edge > spike_window[0]]
    index_first_edge = index_first_edge[index_first_edge < (len(data) - spike_window[1])]
    #print(index_first_edge)

    waveforms = [] #np.zeros(len(index_first_edge), 2 * spike_window[0] + 1);
    pos = [] #np.zeros(len(index_first_edge), 1);
    s=0
    for i in index_first_edge:
        
        index_range = np.arange(i - spike_window[0], i + spike_window[0] )  # waveforms is returned and only used for size
        #print(index_range)
        sig_local = data[index_range]
        peak_index = np.argmax(sig_local)                   # Local index within that array
        new_peak_pos = i - spike_window[0] + (peak_index)   # General index in data
        #print(i - spike_window[0])
        #print(peak_index)
        #print(new_peak_pos)
        #if i == index_first_edge[3]:   
            #sys.exit()
        
        #if new_peak_pos in cardiac_peaks:
            #print(s)
        s=s+1
        #new_peak_pos < (len(data) - spike_window[1]) and \
        if new_peak_pos not in cardiac_peaks:
            if new_peak_pos not in pos:
                print(s)
                pos.append(new_peak_pos)
                #print(index_range[0] + peak_index - 1 - spike_window[1])
                #print(index_range[-1] + peak_index - 1 - spike_window[1])
                new_window = np.arange(new_peak_pos - spike_window[0], new_peak_pos + spike_window[1] )

                #print(new_window)
                #print(new_peak_pos)
                waveforms.append(data[new_window])
                #sys.exit()
    #print(s)    
    #sys.exit()
    wdws = np.stack(np.array(waveforms))
    print('----------------------------------')
    
    return pos, wdws



'''
# Old version AG commented on 06.02.2025
def max_centered_peaks_optimized(data,index_first_edge, spike_window, waveforms=None, noise_peaks=[], verbose=False):
    
    """
    This code is general to center the peaks, no matter if the max or the first cross
    that was initially detected. 

    This function should take in a 1D array data, a list of zero-crossing indices index_first_edge, 
    and an integer spike_window representing the number of samples to extract around each zero-crossing. 
    The function first creates an array of indices spike_indices for the spike window. Then, for each zero-crossing index, 
    the function extracts the spike window around the index from the data, finds the index of the maximum value within the 
    window, calculates the position of the maximum value relative to the start index of the spike window, and stores the 
    position of the maximum value relative to the start index of the data in a list of peak_positions. Finally, the function 
    returns the list of peak positions.

    """
    if verbose:
        print("Aligning peaks to max ")
    index_first_edge = index_first_edge[index_first_edge > spike_window[0]]
    index_first_edge = index_first_edge[index_first_edge < (len(data) - spike_window[1])]
    new_peak_pos = []
    general_index_list = []
    new_peak_amplitude = np.zeros(len(data))
    spikes_vector_loc = np.zeros(len(data))
    s=0

    for i in index_first_edge:
        if i < 4000: #discard first 4000 samples (1-2 sec) for onset artifacts
            continue 
        else:
            index_range = np.arange(i - spike_window[0], i + spike_window[1] + 1)  # waveform is returned and only used for size
                                                                                  # +1 because stop value does not include the value
            #sig_local = data[index_range]
            peak_index = np.argmax(data[index_range])       # Local index within that array

            # General index in data
            general_index = (i - spike_window[0] + (peak_index))   
            
            if data[general_index]<0:
                print('negative peak in final list!')
                continue 
            else:
                # Array of indexes of new window
                gen_sig = np.arange(general_index - spike_window[0], general_index + spike_window[1] + 1)
                # If these indexes are not already in the list, then add the peak as new,
                # otherwie it'll be overlapping with a previous identified peak
                if general_index not in general_index_list:
                    #if data[general_index]<0:
                    #    print('negative peak in final list!')
                    general_index_list.extend(gen_sig)
                    new_peak_pos.append(general_index) 

    unique_peak_pos = set(new_peak_pos) 
    if verbose: 
        print('Number of unique_peak_pos: %s' %len(unique_peak_pos))
    #print('Number of noise_peaks: %s' %len(noise_peaks))

    #pos1 = [x for x in unique_peak_pos if x not in noise_peaks]
    #pos = sorted(pos1)
    pos = sorted(unique_peak_pos.difference(noise_peaks))

    if verbose:
        print('Number of peaks not in cardiac: %s' %len(pos))

    """ No need this here because there is a function just for this get_waveforms
    for p in pos:
        new_window = np.arange(p - spike_window[0], p + spike_window[1] )
        #print(new_window)
        #print(new_peak_pos)
        waveforms.append(data[new_window])

    if waveforms:
        wdws = np.stack(np.array(waveforms))
    else:
        wdws = []
    print('----------------------------------')
    """
    return pos, len(unique_peak_pos), len(noise_peaks)

import numpy as np
'''
def max_centered_peaks_optimized(data, index_first_edge, spike_window, waveforms=None, noise_peaks=set(), verbose=False):
    import numpy as np

def max_centered_peaks_optimized(data, index_first_edge, spike_window, waveforms=None, noise_peaks=[], verbose=False):
    """
    Aligns peaks to the maximum within a spike window around detected zero-crossing indices.

    Parameters:
    - data: 1D numpy array of signal data
    - index_first_edge: numpy array of zero-crossing indices
    - spike_window: tuple (left, right) indicating window size around detected events
    - waveforms: Not used in function but kept for compatibility
    - noise_peaks: List of noise peak indices to exclude
    - verbose: Whether to print debugging info

    Returns:
    - pos: Sorted list of final peak indices (excluding noise)
    - len(unique_peak_pos): Number of unique detected peaks
    - len(noise_peaks): Number of excluded noise peaks
    """
    print('New optimised code')
    if verbose:
        print("Aligning peaks to max...")

    # Remove invalid indices (out of bounds)
    valid_indices = index_first_edge[(index_first_edge > spike_window[0]) & 
                                     (index_first_edge < (len(data) - spike_window[1]))]
    
    # Exclude first 4000 samples (artifact removal)
    valid_indices = valid_indices[valid_indices >= 4000]
    
    # Store detected peaks
    new_peak_pos = []
    general_index_list = set()  # Using a set for fast lookup

    for i in valid_indices:
        # Define the local window around index i
        start_idx = i - spike_window[0]
        end_idx = i + spike_window[1] + 1
        
        index_range = np.arange(start_idx, end_idx)
        
        # Find peak within this window
        peak_index = np.argmax(data[index_range])  
        general_index = start_idx + peak_index  

        # Ensure peak is positive and not already recorded
        if data[general_index] < 0:
            if verbose:
                print(f'Negative peak at {general_index}, skipping...')
            continue

        if general_index not in general_index_list:
            # Store the general index and prevent overlapping peaks
            general_index_list.update(range(start_idx, end_idx))
            new_peak_pos.append(general_index)

    # Remove duplicates while preserving order
    unique_peak_pos = list(dict.fromkeys(new_peak_pos))

    if verbose:
        print(f'Number of unique peaks detected: {len(unique_peak_pos)}')

    # Convert noise_peaks to a set for fast lookup
    noise_peaks_set = set(noise_peaks)
    pos = sorted([x for x in unique_peak_pos if x not in noise_peaks_set])

    if verbose:
        print(f'Number of peaks after noise filtering: {len(pos)}')

    return pos, len(unique_peak_pos), len(noise_peaks)



def return_max_centered_peaks(data,i, spike_window, verbose=False):
    print(i_first_edge)
    if i_first_edge < 4000: #discard first 4000 samples (1-2 sec) for onset artifacts
        print('discard') 
    else:
        index_range = np.arange(i - spike_window[0], i + spike_window[1] + 1)  # waveform is returned and only used for size
                                                                              # +1 because stop value does not include the value
        sig_local = data[index_range]
        peak_index = np.argmax(sig_local)       # Local index within that array

        # General index in data
        general_index = (i - spike_window[0] + (peak_index))   
        
        if data[general_index]<0:
            print('negative peak in final list!')
        else:
            # Array of indexes of new window
            gen_sig = np.arange(general_index - spike_window[0], general_index + spike_window[1] + 1)
            # If these indexes are not already in the list, then add the peak as new,
            # otherwie it'll be overlapping with a previous identified peak
            if general_index not in general_index_list:
                if data[general_index]<0:
                    print('negative peak in final list!')
    print(general_index)
    return [gen_sig, general_index]
    #return gen_sig, general_index

def max_centered_peaks_parallel(data,index_first_edge, spike_window, waveforms=None, noise_peaks=[], verbose=False):
    """
    This code is general to center the peaks, no matter if the max or the first cross
    that was initially detected. 

    TO DO: comment

    """
    tic = time.time()

    num_cores = multiprocessing.cpu_count()
    print('num_cores %s' %num_cores)
    
    if verbose:
        print("Aligning peaks to max ")
    index_first_edge = index_first_edge[index_first_edge > spike_window[0]]
    index_first_edge = index_first_edge[index_first_edge < (len(data) - spike_window[1])]
    new_peak_pos = []
    general_index_list = []
    #pool = multiprocessing.Pool(3)
    #results = pool.starmap(return_max_centered_peaks, product(data,index_first_edge, spike_window))
    #print(results)
    #with multiprocessing.Pool(processes=3) as pool:
        #results = pool.starmap(return_max_centered_peaks, product(data,index_first_edge, spike_window))
    #general_index_list, new_peak_pos = Parallel(n_jobs=3, backend= 'multiprocessing')(delayed(return_max_centered_peaks)(data,i_first_edge, spike_window) for i_first_edge in index_first_edge)
    #zip(*)
    #with parallel_backend('multiprocessing'):

    results = Parallel(n_jobs=num_cores, verbose=1)(delayed(return_max_centered_peaks)(data,i_first_edge, spike_window) for i_first_edge in index_first_edge)
    general_index_list = np.hstack(np.vstack(results[0]))
    new_peak_pos = np.hstack(np.vstack(results[1]))
    print(new_peak_pos)

    unique_peak_pos = set(new_peak_pos) 
    if verbose: 
        print('Number of unique_peak_pos: %s' %len(unique_peak_pos))
    #print('Number of noise_peaks: %s' %len(noise_peaks))

    #pos1 = [x for x in unique_peak_pos if x not in noise_peaks]
    #pos = sorted(pos1)
    pos = sorted(unique_peak_pos.difference(noise_peaks))

    if verbose:
        print('Number of peaks not in cardiac: %s' %len(pos))

    return pos, len(unique_peak_pos), len(noise_peaks)


def min_centered_peaks_optimized(data,index_first_edge, spike_window, waveforms=None, noise_peaks=[], verbose=False):
    """
    This code is general to center the peaks around the min value (adapted from above).

    TO DO: comment

    """
    if verbose:
        print("Aligning peaks to min")
    index_first_edge = index_first_edge[index_first_edge > spike_window[0]]
    index_first_edge = index_first_edge[index_first_edge < (len(data) - spike_window[1])]
    new_peak_pos = []
    general_index_list = []
    new_peak_amplitude = np.zeros(len(data))
    spikes_vector_loc = np.zeros(len(data))
    s=0

    for i in index_first_edge:
        if i < 4000: #discard first 4000 samples (1-2 sec) for onset artifacts
            continue 
        else:
            index_range = np.arange(i - spike_window[0], i + spike_window[1] + 1)  # waveform is returned and only used for size
                                                                                  # +1 because stop value does not include the value
            sig_local = data[index_range]
            peak_index = np.argmin(sig_local)       # Local index within that array

            # General index in data
            general_index = (i - spike_window[0] + (peak_index))   
            
            if data[general_index]>0:
                print('positive peak in final list!')
                continue 
            else:
                # Array of indexes of new window
                gen_sig = np.arange(general_index - spike_window[0], general_index + spike_window[1] + 1)
                # If these indexes are not already in the list, then add the peak as new,
                # otherwise it'll be overlapping with a previous identified peak
                if general_index not in general_index_list:
                    if data[general_index]>0:
                        print('positive peak in final list!')
                    general_index_list.extend(gen_sig)
                    new_peak_pos.append(general_index) 

    unique_peak_pos = set(new_peak_pos) 
    if verbose: 
        print('Number of unique_peak_pos: %s' %len(unique_peak_pos))
    #print('Number of noise_peaks: %s' %len(noise_peaks))

    #pos1 = [x for x in unique_peak_pos if x not in noise_peaks]
    #pos = sorted(pos1)
    pos = sorted(unique_peak_pos.difference(noise_peaks))

    if verbose:
        print('Number of peaks not in cardiac: %s' %len(pos))

    return pos, len(unique_peak_pos), len(noise_peaks)

def k_means(data, num_clus=3, steps=200):

    # Convert data to Numpy array
    cluster_data = np.array(data)

    # Initialize by randomly selecting points in the data
    center_init = np.random.randint(0, cluster_data.shape[0], num_clus)

    # Create a list with center coordinates
    center_init = cluster_data[center_init, :]

    # Repeat clustering  x times
    for _ in range(steps):

        # Calculate distance of each data point to cluster center
        distance = []
        for center in center_init:
            tmp_distance = np.sqrt(np.sum((cluster_data - center)**2, axis=1))
            
            # Adding smalle random noise to the data to avoid matching distances to centroids
            tmp_distance = tmp_distance + np.abs(np.random.randn(len(tmp_distance))*0.0001)
            distance.append(tmp_distance)

        # Assign each point to cluster based on minimum distance
        _, cluster = np.where(np.transpose(distance == np.min(distance, axis=0)))

        # Find center of mass for each cluster
        center_init = []
        for i in range(num_clus):    
            center_init.append(cluster_data[cluster == i, :].mean(axis=0).tolist())
            
    return cluster, center_init, distance


def clustering_avg_dis(pca_components, max_num_clusters=15):
    average_distance = []
    for run in range(20):
        tmp_average_distance = []
        for num_clus in range(1, max_num_clusters + 1):
            cluster, centers, distance = k_means(pca_components, num_clus)
            tmp_average_distance.append(np.mean(
                [np.mean(distance[x][cluster == x]) for x in range(num_clus)],
                axis=0))
        average_distance.append(tmp_average_distance)

    return average_distance

def get_spike_amplitude(signal, spikes_idx=[]):
    # Moved to utils as it's a general method
    spikes_vector = np.zeros(len(signal))
    if type(spikes_idx) == 'list':
        spikes_idx_array = np.array(list(spikes_idx))
        sys.exit()
        if spikes_idx:
            spikes_vector[spikes_idx_array] = signal[spikes_idx_array]
        else:
            spikes_vector = []
    else: # It's an array
        if spikes_idx != []:
            spikes_vector[spikes_idx] = signal.iloc[spikes_idx]
        else:
            spikes_vector = []
        
    return spikes_vector   

def NEO(data):
    neoData = np.zeros(len(data))

    for i in np.arange(1, len(data) - 1):
        neoData[i - 1] = np.power(data[i], 2) - data[i - 1] * data[i + 1]
    """
    plt.plot(neoData)
    plt.show()
    plt.close('all')
    """
    return neoData


def NEOthreshold(neosg, fs, C=8):
    th = C * np.mean(neosg)
    print(th)
    # neosg[neosg < th] = 0

    # Find positions wherere the threshold is crossed
    pos, peaks_options = signal.find_peaks(neosg, height=[th, ],
                                                  distance=0.005 * fs)
    # Create peaks vector
    neo_peaks = np.zeros(neosg.shape)
    neo_peaks[pos] = th
    return neo_peaks, pos


def waveform(data, peaks, fs, filtData):
    spikeData=np.zeros(len(data))
    spikeData[peaks]=1
    plt.plot(spikeData)
    plt.show()
    print(np.floor(3*fs*0.001+1))
    print(np.floor(fs*0.001))
    print(np.floor(len(spikeData)-fs*0.002))
    print(np.sum(spikeData[int(np.floor(fs*0.001)):int(np.floor(len(spikeData)-fs*0.002))]))
    #sys.exit()
    spikes=np.zeros(int(np.sum(spikeData[int(np.floor(fs*0.001)):int(np.floor(len(spikeData)-fs*0.002))])))
    count=0

    print(np.floor(fs*0.001))
    print(np.floor(len(data)-fs*0.002))
    for i,k in np.enumerate(np.linspace(np.floor(fs*0.001), np.floor(len(spikeData)-fs*0.002))):
        print(i)
        if spikeData[i] == 1:
            count = count + 1
            spikes[count,:] = filtData[int(i - np.floor(fs * 0.001)):int(i + np.floor(fs * 0.002))]
            indStart[count] = i - np.floor(fs * 0.001)
            plt.plot(spikes[count,:])
            plt.show()
def movmean(T, m):
    assert(m <= T.shape[0])
    n = T.shape[0]
 
    sums = np.zeros(n - m + 1)

    sums[0] = np.sum(T[0:m-1])
    
    cumsum = np.cumsum(T)
    cumsum = np.insert(cumsum, 0, 0) # Insert a 0 at the beginning of the array
    
    sums = cumsum[m:] - cumsum[:-m]

    return sums/m

def movstd(T, m):
    n = T.shape[0]
    
    cumsum = np.cumsum(T)
    cumsum_square = np.cumsum(T**2)
    
    cumsum = np.insert(cumsum, 0, 0)               # Insert a 0 at the beginning of the array
    cumsum_square = np.insert(cumsum_square, 0, 0) # Insert a 0 at the beginning of the array
    
    seg_sum = cumsum[m:] - cumsum[:-m]
    seg_sum_square = cumsum_square[m:] - cumsum_square[:-m]
    
    return np.sqrt( seg_sum_square/m - (seg_sum/m)**2 )

def so_cfar_Zanos(y, w, g, windowmode='socfar', nstd= [1, 1], verbose=False):
    """
    Adapted from Zanos Matlab code. Checked implementation with his code and same results
    However, don't think it does what's supposed to do the SO-CFAR (Bad implementation by Zanos?)
    """
    # Apply an adaptive threshold.
    # Compute a rolling std in each window, and choose the side that has
    # the smallest value. w are the windows, g are the guard regions, and
    # CUT is the cell under test.
    #
    # |----w-----|--g--|CUT|--g--|----w-----|
    #
    # y           - neural signal
    # w           - width of the CFAR window on either side in samples
    # g           - width of the guard region on either side in samples
    # windowmode  - 'rollingstd' (single window) or 'socfar'
    # plotflag    - display a plot
    # nstd        - number of standard deviations on each side of the mean
    #               to apply the threshold
    # plotinds    - indices to display in the plot
    # 
    # indspos     - indices that exceed the positive threshold
    # indsneg     - indices that exceed the negative threshold
    
    if verbose:
        print("Starting SO-CFAR....")

    time_start = time.time()

    # the window length must be odd. #if condition returns False, AssertionError is raised:
    assert (w % 2) != 0, 'w must be odd'

    if np.isscalar(nstd):
        nstd = np.repmat(nstd, [1, 2])
    
    
   
    # rolling mean and rolling std
    #rollingstd = movstd(y, w);
    #rollingmean = movmean(y, w);
    rollingstd = y.rolling(w, min_periods=2, center=True).std()
    rollingmean = y.rolling(w, min_periods=2, center=True).mean()
   
    if windowmode == 'socfar':
        # pair leading and lagging windows
        cfarwindowsstd = np.vstack((np.array(rollingstd[:-(w + 2 * g )]).ravel(),
                                    np.array(rollingstd[w + 2 * g :len(rollingstd)]).ravel())).T
        cfarwindowsmean = np.vstack((np.array(rollingmean[:-(w + 2 * g )]).ravel(),
                                    np.array(rollingmean[w + 2 * g :len(rollingmean)]).ravel())).T
        mv = cfarwindowsstd.min(axis=1) #Axis 2 in matlab
        mi = np.argmin(cfarwindowsstd, axis=1)
        # select corresponding mean
        mean2 = np.zeros(len(y))
        for x in np.arange(len(mi)):
            mean2[int((w - 1) / 2 + g + 1 + x)] = cfarwindowsmean[x, mi[x]]
        #print(sum(mean2))
        # selected std with zero padding
        yout = np.append(np.zeros(int((w - 1) / 2 + g + 1)), mv)
        yout = np.append(yout, np.zeros(int((w - 1) / 2 + g)))
        #print(sum(yout))
    else:
        yout = rollingstd
        mean2 = rollingmean
    
    """
    # plot
    if plotflag        
        figure;
        plot(t(plotinds), [y(plotinds) - mean2(plotinds), ... # AG was reduce_plot
                      nstd(1) * yout(plotinds), ...
                      -nstd(2) * yout(plotinds)]);
        xlabel('Time (seconds)');
    end
    """
    # find threshold crossings
    thresh = np.vstack((nstd[0] * yout + mean2, -nstd[1] * yout + mean2)).T
    indspos = np.where(np.array(y).ravel() > thresh[:, 0])
    indsneg = np.where(np.array(y).ravel() < thresh[:, 1])
    
    print("SO_CFAR finished! Time elapsed: {} seconds".format(time.time()-time_start))
    return indspos[0], indsneg[0], thresh

def detect_peaks_CFAR(data, spike_window, num_train, num_guard, rate_fa):
    """
    DO NOT USE! DOESN"T WORK!
    Detect peaks with CFAR algorithm.
    From: https://tsaith.github.io/detect-peaks-with-cfar-algorithm.html


    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate. 
    """
    print("Starting detect_peaks_CFAR...")
    num_cells = data.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half
 
    alpha = num_train*(rate_fa**(-1/num_train) - 1) # threshold factor
    
    peak_idx = []
    waveforms = []
    thresh = np.zeros(num_cells)
    a= 0
    b=0
    print(num_side)
    print(num_cells)
    for i in range(num_side, num_cells - num_side):
        
        if i == i-num_side+np.argmax(data[i-num_side:i+num_side+1]): 
            b=b+1
            continue
        
        sum1 = np.sum(data[i-num_side:i+num_side+1])
        sum2 = np.sum(data[i-num_guard_half:i+num_guard_half+1]) 
        p_noise = (sum1 - sum2) / num_train 
        threshold = alpha * p_noise
        thresh[i] = threshold
        a = a+1
        if data[i] > threshold: 
            peak_idx.append(i)
            new_window = np.arange(i - spike_window[0], i + spike_window[1] )

            #print(new_window)
            #print(new_peak_pos)
            waveforms.append(data[new_window])
            #sys.exit()
    wdws = np.stack(np.array(waveforms))
    peak_idx = np.array(peak_idx, dtype=int)
    print(a)
    print(b)
    print("Finishing detect_peaks_CFAR")
    return peak_idx, wdws, thresh


def get_waveforms(data, spikes_idx, spike_window):
    waveforms = []
    for i in spikes_idx: 
        new_window = np.arange(i - spike_window[0], i + spike_window[1] )
        waveforms.append(data[new_window])
        #sys.exit()
    wave_form = np.stack(np.array(waveforms))
    return wave_form


def wavelet_decomp(y, fs, type='dwt',verbose=False, **kwargs):
    """
    Wavelet decomposition

    Parameters
    --------------
    y:      [array_like] Input data
    fs:     [float] sampling frequency of data
    type:   [string] Type of wavelet analysis to performs. By defauls MRA based on dwt
            SWT: Stationary Wavelet Transform (SWT), also known as Undecimated wavelet transform or Algorithme à trous is a 
            translation-invariance modification of the Discrete Wavelet Transform that does not decimate coefficients at every transformation level.
    verbose:[int - 0,1] signal to display text information (default 1 - show text, 0 - don't show) 
    **kwargs: specific parameters:
        wavelet: Wavelet object or name string. Wavelet to use.
        level: [int, optional] Decomposition level (must be >= 0). If level is None (default) then it will be calculated using the dwt_max_level function.

    Other parameters and intermediate returns
    -----------------------------------------
    coeffs : array_like. Coefficients list [cAn, cDn, cDn-1, …, cD2, cD1]
    """
    if verbose:
        print("Wavelet decomposition started...")    
    if type=="dwt":
        # Multilevel decomposition using wavedec
        coeffs = pywt.wavedec(y, wavelet=kwargs['wavelet'], level=kwargs['level'])
        st = kwargs['start_level']
        end = kwargs['end_level']
        #coeffs[0] = np.zeros_like(coeffs[0]) # Approx coeff to zero
        for i in np.arange(st, end):
            coeffs[i] = np.zeros_like(coeffs[i])
        # Multilevel reconstruction using waverec: Multilevel 1D Inverse Discrete Wavelet Transform.
        sig = pywt.waverec(coeffs, wavelet=kwargs['wavelet'])
        
    elif type=='swt':
        ## Multilevel 1D stationary wavelet transform.
        coeffs = pywt.swt(y, axis=-1, **kwargs)
        sig = pywt.iswt(coeffs, wavelet=kwargs['wavelet'])
    else:
        ## CWT tranform
        # OJO: db no esta disponible para cwt
        #print(kwargs)
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        if kwargs['twave']>0:
            # Before: coeffs, freq = pywt.cwt(y, [kwargs['twave']*fs-0.001, kwargs['twave']*fs+0.001], wavelet=kwargs['wavelet'])
            coeffs, freq = pywt.cwt(y, [(kwargs['twave']-kwargs['tinterval'])*fs, (kwargs['twave']+kwargs['tinterval'])*fs], wavelet=kwargs['wavelet'])
        elif np.mean(kwargs['scales'])!=0:
            print('scales')
            coeffs, freq = pywt.cwt(y, kwargs['scales'], wavelet=kwargs['wavelet'])
        #print(coeffs)
        sig = np.mean(coeffs, 0)
    if verbose:
        print("Wavelet decomposition finished")
    sig = sig[:len(y)] #Ensure we're getting the same output length
    return sig

def DWT_with_denoising(y, fs, verbose=False, **kwargs):
    """
    Based on the work for adaptive filtering of Diedrich et al and 2003 Gao et al. 2010
    
    The regular wavelet de-noising technique proposed by Donoho [15], [16] includes a 
    threshold for each decomposition level:
    T = σ sqrt(2ln N) (1A)
    where σ is the standard deviation of the Gaussian noise and N is the number of samples in
    the signal. The σ for each level is estimated as the median absolute value of the wavelet
    coefficients divided by 0.6745, which is the 75th percentile of the standard normal
    distribution [15], [16]. 

    Donoho then used detail coefficients which are modified by soft
    thresholding (2A) for reconstruction of the de-noised signal [15], [16].
    y =  sign(x) (abs(x)− T) , if abs(x) > T 0,    
      =  0,                     if abs(x) ≤ Τ


    hard thresholding instead of soft thresholding (2B)
    y = x if abs(x) > T 
      = 0 if abs(x) ≤ T .
    """
    if verbose:
        print("DWT_with_denoising started...")    
    coeffs = pywt.wavedec(y, wavelet=kwargs['wavelet'], level=kwargs['level'])
    st = kwargs['start_level']
    end = kwargs['end_level']

    #new_coeffs = np.zeros_like(coeffs)
    new_coeffs = []
    old_coeffs = np.zeros_like(coeffs)
    thres = np.zeros(kwargs['level']+1)

    #coeffs[0] = np.zeros_like(coeffs[0]) # Approx coeff to zero
    for i in np.arange(st, end):
        coeffs[i] = np.zeros_like(coeffs[i])
    
    if kwargs['DBplot']:
        fig, axes = plt.subplots(nrows=kwargs['level']+1, ncols=1, clear=True, sharex=False)

    # Threshold each coefficient
    for i in np.arange(0, kwargs['level']+1):
        #print(len(coeffs[i]))
        # Compute threshold value
        sigma = stats.median_abs_deviation(coeffs[i])/0.6745

        th = kwargs['k']*sigma*np.sqrt(2*np.log(len(coeffs[i])))
        #print(th)
        sig_th = coeffs[i]
        old_coeffs[i] =  sig_th
        
        # Get indices above and below threshold
        below_th_ind = np.where(np.abs(sig_th) < th)[0]
        above_th_inds = np.where(np.abs(sig_th) >= th)[0]
        if kwargs['DBplot']:
            axes[i].plot(coeffs[i], '-', linewidth=1.5, label='Original')
        # Select threshold type
        if kwargs['thres_type']=="hard":
            sig_th[below_th_ind] = 0
        
        elif kwargs['thres_type']=="soft":
            sig_th[below_th_ind] = 0
            sig_th[above_th_inds] = np.sign(sig_th[above_th_inds])*(abs(sig_th[above_th_inds])-th)

        new_coeffs.append(sig_th)
        thres[i] = th
        if kwargs['DBplot']:
            axes[i].plot(new_coeffs[i], '-', linewidth=0.5, label='De-noised')
            axes[i].plot(thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5, label='th')
            axes[i].plot(-thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5, label='-th')
            axes[i].legend(loc='upper right')
    # plt.show()

    #print(sum(new_coeffs[-1]))
    #print(sum(new_coeffs[-1]-sig_th))
    
    
    """
    # For some reason coeffs becomes new_coeffs and the plot is the same...
    if DBplot:
        fig, axes = plt.subplots(nrows=kwargs['level'], ncols=1, clear=True, sharex=False)
        for i in np.arange(1, kwargs['level']):
            axes[i].plot(coeffs[i], '-', linewidth=1.5, label='Original')
            axes[i].plot(new_coeffs[i], '-', linewidth=0.5, label='De-noised')
            axes[i].plot(thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5)
            axes[i].plot(-thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5)
            axes[i].legend(loc='upper right')
        plt.show()
    """
    # Reconstruct signal
    sig = pywt.waverec(new_coeffs, wavelet=kwargs['wavelet'])
    sig = sig[:len(y)] #Ensure we're getting the same output length
    if verbose:
        print("Wavelet decomposition finished")
    return sig

def SWT_with_denoising(y, fs, k=1,thres_type="soft", DBplot=False, **kwargs):
    """
    Based on the work for adaptive filtering of Diedrich et al and 2003 Gao et al. 2010
    
    The regular wavelet de-noising technique proposed by Donoho [15], [16] includes a 
    threshold for each decomposition level:
    T = σ sqrt(2ln N) (1A)
    where σ is the standard deviation of the Gaussian noise and N is the number of samples in
    the signal. The σ for each level is estimated as the median absolute value of the wavelet
    coefficients divided by 0.6745, which is the 75th percentile of the standard normal
    distribution [15], [16]. 

    Donoho then used detail coefficients which are modified by soft
    thresholding (2A) for reconstruction of the de-noised signal [15], [16].
    y =  sign(x) (abs(x)− T) , if abs(x) > T 0,    
      =  0,                     if abs(x) ≤ Τ


    hard thresholding instead of soft thresholding (2B)
    y = x if abs(x) > T 
      = 0 if abs(x) ≤ T .
    """
    print("SWT_with_denoising started...")    
    coeffs = pywt.swt(y, axis=-1, **kwargs)
    num_dec_levels = len(coeffs[0])
    #print(num_dec_levels)
    new_coeffs = []
    thres = np.zeros(num_dec_levels)
        
    fig, axes = plt.subplots(nrows=num_dec_levels, ncols=1, clear=True, sharex=False)

    # Threshold each coefficient
    for i in np.arange(0, num_dec_levels):
        #print(i)
        # Compute threshold value
        sigma = stats.median_absolute_deviation(coeffs[0][i])/0.6745
        th = k*sigma*np.sqrt(2*np.log(len(coeffs[0][i])))
        #print(th)
        sig_th = coeffs[0][i]

        # Get indices above and below threshold
        below_th_ind = np.where(np.abs(sig_th) < th)[0]
        above_th_inds = np.where(np.abs(sig_th) >= th)[0]
        axes[i].plot(coeffs[0][i], '-', linewidth=1.5, label='Original')
        # Select threshold type
        if thres_type=="hard":
            sig_th[below_th_ind] = 0
        
        elif thres_type=="soft":
            sig_th[below_th_ind] = 0
            sig_th[above_th_inds] = np.sign(sig_th[above_th_inds])*(abs(sig_th[above_th_inds])-th)


        new_coeffs.append(sig_th)
        thres[i] = th
        axes[i].plot(new_coeffs[i], '-', linewidth=0.5, label='De-noised')
        axes[i].plot(thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5, label='th')
        axes[i].plot(-thres[i]*np.ones(len(new_coeffs[i])), '-', linewidth=0.5, label='-th')
        axes[i].legend(loc='upper right')
    #plt.show()

    
    # Reconstruct signal
    sig = pywt.iswt(new_coeffs, wavelet=kwargs['wavelet'])

    print("Wavelet decomposition finished")
    return sig




def gen_nleo(x, l=1, p=2, q=0, s=3):
    """general form of the nonlinear energy operator (NLEO)
    General NLEO expression: Ψ(n) = x(n-l)x(n-p) - x(n-q)x(n-s)
    for l+p=q+s  (and [l,p]≠[q,s], otherwise Ψ(n)=0)

    https://github.com/otoolej/envelope_derivative_operator/blob/1b3395f6f36f32084e65d37a2bba5323b18320c0/energy_operators/general_nleo.py

    Parameters
    ----------
    x: ndarray
        input signal
    l: int, optional
        parameter of NLEO expression (see above)
    p: int, optional
        parameter of NLEO expression (see above)
    q: int, optional
        parameter of NLEO expression (see above)
    s: int, optional
        parameter of NLEO expression (see above)
    Returns
    -------
    x_nleo : ndarray
        NLEO array
    Example
    -------
    import numpy as np
    # generate test signal
    N = 256
    n = np.arange(N)
    w1 = np.pi / (N / 32)
    ph1 = -np.pi + 2 * np.pi * np.random.rand(1)
    a1 = 1.3
    x1 = a1 * np.cos(w1 * n + ph1)
    # compute instantaneous energy:
    x_nleo = gen_nleo(x1, 1, 2, 0, 3)
    # plot:
    plt.figure(1, clear=True)
    plt.plot(x1, '-o', label='test signal')
    plt.plot(x_nleo, '-o', label='Agarwal-Gotman')
    plt.legend(loc='upper left')
    """
    # check parameters:
    if ((l + p) != (q + s) and any(np.sort((l, p)) != np.sort((q, s)))):
        warning('Incorrect parameters for NLEO. May be zero!')

    N = len(x)
    x_nleo = np.zeros(N)

    iedges = abs(l) + abs(p) + abs(q) + abs(s)
    n = np.arange(iedges + 1, (N - iedges - 1))

    x_nleo[n] = x[n-l] * x[n-p] - x[n-q] * x[n-s]

    return(x_nleo)


def specific_nleo(x, type='teager'):
    """ generate different NLEOs based on the same operator 
    Parameters
    ----------
    x: ndarray
        input signal
    type: {'teager', 'agarwal', 'palmu', 'abs_teager', 'env_only'}
        which type of NLEO? 
    Returns
    -------
    x_nleo : ndarray
        NLEO array
    """

    def teager():
        return(gen_nleo(x, 0, 0, 1, -1))

    def agarwal():
        return(gen_nleo(x, 1, 2, 0, 3))

    def palmu():
        return(abs(gen_nleo(x, 1, 2, 0, 3)))

    def abs_teager():
        return(abs(gen_nleo(x, 0, 0, 1, -1)))

    def env_only():
        return(abs(x) ** 2)

    def default_nleo():
        # -------------------------------------------------------------------
        # default option
        # -------------------------------------------------------------------
        print('Invalid NLEO name; defaulting to Teager')
        return(teager())

    # pick which function to execute
    which_nleo = {'teager': teager, 'agarwal': agarwal,
                  'palmu': palmu, 'abs_teager': abs_teager,
                  'env_only': env_only}

    def get_nleo(name):
        return which_nleo.get(name, default_nleo)()

    x_nleo = get_nleo(type)
    return(x_nleo)


def test_compare_nleos_and_NDO(x=None, DBplot=True):
    """ test all NLEO variants with 1 signal
    Parameters
    ----------
    x: ndarray, optional
        input signal (defaults to coloured Gaussian noise)
    DBplot: bool
        plot or not
    """
    if x is None:
        N = 128
        x = np.cumsum(np.random.randn(N))

    all_methods = ['teager', 'agarwal', 'palmu']
    all_methods_strs = {'teager': 'Teager-Kaiser', 'agarwal': 'Agarwal-Gotman',
                        'palmu': 'Palmu et.al.'}
    x_nleo = dict.fromkeys(all_methods)
    x_e = gen_edo(x)

    for n in all_methods:
        x_nleo[n] = specific_nleo(x, n)

    if DBplot:
        fig, ax = plt.subplots(nrows=3, ncols=1, num=4, clear=True, sharex=True)
        ax[0].plot(x, '-', linewidth=0.5, label='test signal')
        for n in all_methods:
            ax[1].plot(x_nleo[n], '-', linewidth=0.5, label=all_methods_strs[n])
        ax[2].plot(x_e, '-', linewidth=0.5, label='EDO')
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper left')
        ax[2].legend(loc='upper left')

        #plt.pause(60.0001)
        plt.show()




def gen_edo(x, DBplot=False):
    """compute the envelope derivative operator (EDO), as defined in [1].
    [1] JM O' Toole, A Temko, NJ Stevenson, “Assessing instantaneous energy in the EEG: a
    non-negative, frequency-weighted energy operator”, IEEE Int. Conf.  on Eng. in Medicine
    and Biology, Chicago, August 2014
    John M. O' Toole, University College Cork
    Started: 05-09-2019
    last update: <2019-09-04 13:36:01 (otoolej)
    https://github.com/otoolej/envelope_derivative_operator/blob/1b3395f6f36f32084e65d37a2bba5323b18320c0/energy_operators/edo.py
    >
    """

    """Generate EDO Γ[x(n)] from simple formula in the time domain:
    Γ[x(n)] = y(n)² + H[y(n)]²
    where y(n) is the derivative of x(n) using the central-finite method and H[.] is the
    Hilbert transform.
    Parameters
    ----------
    x: ndarray
        input signal
    DBplot: bool, optional
        plot or not
    Returns
    -------
    x_edo : ndarray
        EDO of x
    """
    # 1. check if odd length and if so make even:
    N_start = len(x)
    if (N_start % 2) != 0:
        x = np.hstack((x, 0))

    N = len(x)
    nl = np.arange(1, N - 1)
    xx = np.zeros(N)

    # 2. calculate the Hilbert transform
    h = discrete_hilbert(x)

    # 3. implement with the central finite difference equation
    xx[nl] = ((x[nl+1] ** 2) + (x[nl-1] ** 2) +
              (h[nl+1] ** 2) + (h[nl-1] ** 2)) / 4 - ((x[nl+1] * x[nl-1] +
                                                       h[nl+1] * h[nl-1]) / 2)

    # trim and zero-pad and the ends:
    x_edo = np.pad(xx[2:(len(xx) - 2)], (2, 2),
                   'constant', constant_values=(0, 0))

    return(x_edo[0:N_start])

def discrete_hilbert(x, DBplot=False):
    """Discrete Hilbert transform
    Parameters
    ----------
    x: ndarray
        input signal
    DBplot: bool, optional
        plot or not 
    Returns
    -------
    x_hilb : ndarray
        Hilbert transform of x
    """
    N = len(x)
    Nh = np.ceil(N / 2)
    k = np.arange(N)

    # build the Hilbert transform in the frequency domain:
    H = -1j * np.sign(Nh - k) * np.sign(k)
    x_hilb = np.fft.ifft(np.fft.fft(x) * H)
    x_hilb = np.real(x_hilb)

    if DBplot:
        plt.figure(10, clear=True)
        plt.plot(np.imag(H))

    return(x_hilb)

def test_edo_random(x=None, DBplot = True):
    """test EDO with a random signal"""
    # if (__name__ == '__main__'):

    if x is None:
        x = np.random.randn(102)
    x_e = gen_edo(x)

    # -------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------
    if DBplot:
        fig, ax = plt.subplots(nrows=2, ncols=1, clear=True, sharex=True)
        ax[0].plot(x, '-', linewidth=0.5, label='test signal')
        ax[1].plot(x_e, '-', linewidth=0.5, label='EDO')
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper left')
        plt.pause(60.0001)
        # plt.show(block=False)


def linear_envelope(x, sampling_rate=40000, freqs=[10, 400], lfreq=4, DBplot=False):
    r"""Calculate the linear envelope of a signal.

    Parameters
    ----------
    x : array
        raw signal.
    sampling_rate : int
        Sampling rate (samples/second).
    freqs : list [fc_h, fc_l], optional
            cutoff frequencies for the band-pass filter (in Hz).
    lfreq : number, optional
            cutoff frequency for the low-pass filter (in Hz).

    Returns
    -------
    envelope : array
        linear envelope of the signal.

    Notes
    -----

    *Authors*

    - Marcos Duarte
    https://neurokit.readthedocs.io/en/latest/_modules/neurokit/bio/bio_emg.html

    *See Also*

    See this notebook [1]_.

    References
    ----------
    .. [1] https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
    """
    x_t = gen_nleo(x, 0, 0, 1, -1)

    if np.size(freqs) == 2:
        # band-pass filter
        b, a = sp.signal.butter(2, np.array(freqs)/(sampling_rate/2.), btype = 'bandpass')
        x_t = sp.signal.filtfilt(b, a, x_t)
    if np.size(lfreq) == 1:
        # full-wave rectification
        envelope = abs(x_t)
        # low-pass Butterworth filter
        b, a = sp.signal.butter(2, np.array(lfreq)/(sampling_rate/2.), btype = 'low')
        envelope = sp.signal.filtfilt(b, a, envelope)

        if DBplot:
            fig, ax = plt.subplots(nrows=3, ncols=1, clear=True, sharex=True)
            ax[0].plot(x, '-', linewidth=0.5, label='signal')
            ax[1].plot(x_t, '-', linewidth=0.5, label='TEO signal')
            ax[2].plot(envelope, '-', linewidth=0.5, label='Envelope')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper left')
            ax[2].legend(loc='upper left')
            plt.show()


    return (envelope)

def check_MRA_wavelet_decomp(signal, fs, check='level', mra_type='dwt', wvl='db3', st=1, end=8, l=8):

    if check=='level':
        """Checking levels MRA reconstruction"""
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        axes = axes.ravel()
        if mra_type=="dwt":
            for i in np.arange(st,l):
                print(i)  
                dec_signal = wavelet_decomp(signal, fs, type=mra_type,
                                            wavelet=wvl, level=l,
                                            start_level=st, end_level=(l+1-i))
                axes[i-1].plot(signal.to_numpy(), '-', linewidth=0.5, label='Original')
                axes[i-1].plot(dec_signal, '-', linewidth=0.5, label='%s recostructed' %mra_type)
                axes[i-1].set_xlabel('%s to %s' %(st, l+1-i))
        else:
            for i in np.arange(0,l):
                print(i)
                try:
                    dec_signal = wavelet_decomp(neural_sig, fs, type='swt',
                                                wavelet=wvl, level=l,
                                                start_level=i)
                    lev=l
                except:
                    print('level too high')
                    dec_signal = wavelet_decomp(neural_sig, fs, type='swt',
                                                wavelet=wvl, level=(l-i),
                                                start_level=i)
                    lev=l-i
                axes[i].plot(signal, '-', linewidth=0.5, label='Original')
                axes[i].plot(dec_signal, '-', linewidth=0.5, label='%s recostructed' %mra_type)
                axes[i].set_xlabel('Starting: %s, Number of levels: %s' %(i, lev))
                axes[i].legend(loc='upper right')
    elif check=='wavelet':    
        """Checking db family MRA reconstruction"""

        fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
        axes = axes.ravel()
        waveform = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10']
        #waveform = ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10']
        for i, elem in enumerate(waveform):
            print(elem)
            neural_wvl = wavelet_decomp(signal, fs, type='dwt',
                                        wavelet=elem, level=l,
                                        start_level=st, end_level=end)
            axes[i].plot(signal.to_numpy(),'-', linewidth=0.5, label='Original')
            axes[i].plot(neural_wvl,'-', linewidth=0.5, label='%s recostructed' %elem)
            axes[i].set_xlabel('%s' %elem)
        
        #plt.show()
    return

'''
Functions that extracted the features listed above from the signal. 
All these functions take a pandas series as input and usually output a number as float. 
Some of these functions also have optional inputs. 

List of all features: [ Mean , Max , Min , MAV , Var , StD , WL , Energy_from_fft , Kurtosis , Skewness ,\
                        Signal_Power , Signal_Energy , Min_max_difference , Zero_crossing , Slope_sign_changes, \
                       Wilson_Amplitude , Root_Mean_Square , V3  , Log_detector , DABS , Maximum_fractal_length ,\
                      Myopulse_percentage_rate , Mean_Frequency ]

Some of these features might only work well with reasonably large window. The error is roughly in the order of 1/N percent, due to some common
approximation (sometimes implicit) such as N-1 ~ N. This shouldn't be a big issue when the window length is bigger then 0.1s, for example.
'''

# Feature 0 : General description
def signal_description(signal_series):
    print(signal_series.describe())
    return signal_series.describe()

def Mean(signal_series):
    mean = signal_series.mean()
    #print(f'mean:{mean}')
    return mean

def Max(signal_series):
    maximum = signal_series.max()
    #print(f'Maximum: {maximum}')
    return maximum

def Min(signal_series):
    minimum = signal_series.min()
    #print(f'Minimum: {minimum}')
    return minimum


# Feature 1 : MAV (mean absolute value)
def MAV(signal_series):
    signal_series_abs = signal_series.abs()
    MAV = signal_series_abs.mean()
    #print(f'Mean absolute value (MAV): {MAV}')
    return MAV


# Feature 2 : Variance of the signal. 
# (Feature like variance is often considered as a power-related feature)
def Var(signal_series):
    Var = signal_series.var()
    #print(f'Variance (Var): {Var}')
    return Var


# Feature 3 : Standard deviation of the signal.
def StD(signal_series):
    StD = signal_series.std()
    #print(f'Standard deviation (StD): {StD}')
    return StD


# Feature 4: Waveform length (WL)
'''
NOT to confused with wavelength.
Documented in article "Control of Multifunctional Prosthetic Hands by Processing the Electromyographic Signal" by M. Zecca in more detail.

Also, "Research On the identification of sensory information from mixed nerves by using single-channel cuff
electrodes" by Raspopovic provides very good information on the feature as well
'''

def WL(signal_series):
    signal_series_diff_abs = signal_series.diff().abs()
    WL = signal_series_diff_abs.sum()
    #print(f'Waveform Length (WL): {WL}')
    return WL


# Feature 5: Energy calculated based on Discrete Fourier Transform (DFT). (With fast fourier transform algorithm FFT
def Energy_from_fft(signal_series):
    signal_array = signal_series.to_numpy()
    # Exclude NaN values from the signal array
    signal_array = signal_array[~np.isnan(signal_array)]
    fft = sp.fft.fft(signal_array)
    fft_modulus = np.abs(fft)
    fft_modulus_square = np.square(fft_modulus)
    Energy_from_fft = fft_modulus_square.mean()
    #print(f'Energy_from_fft: {Energy_from_fft}')
    return Energy_from_fft


#  Feature 6: Kurtosis. For control, normal distribution should yield a kurtosis of 0. 
def Kurtosis(signal_series):
    signal_array = signal_series.to_numpy()
    Kurtosis = sp.stats.kurtosis(signal_array, nan_policy='omit')
    #print(f'Kurtosis: {Kurtosis}')
    return Kurtosis


# Feature 7: Skewness (The sample skewness is computed as the Fisher-Pearson coefficient of skewness)
def Skewness(signal_series):
    signal_array = signal_series.to_numpy()
    Skewness = sp.stats.skew(signal_array, nan_policy='omit')
    #print(f'Skewness: {Skewness}')
    return Skewness

'''
The two following features, Power and energy of a signal is cited 
from: https://www.gaussianwaves.com/2013/12/power-and-energy-of-a-signal/

There are various definitions of Power and Energy of the signal. The following approaches use the mean of square and sum of
square to represent signal power and energy. This approach does not take sampling frequency into account (default to 1).

Consequently, we would expect that Signal_energy/Signal power = number of sample data.
'''

# Feature 8: Signal Power (directly calculated from the size of signal), equivalent to Mean Square
def Signal_Power(signal_series):
    # Exclude NaN values from the signal series
    signal_array = signal_series.to_numpy()
    signal_array = signal_array[~np.isnan(signal_array)]
    signal_array_abs = np.abs(signal_array)
    signal_array_abs_square = np.square(signal_array_abs)
    Signal_power = np.mean(signal_array_abs_square)
    # print(f'Signal_power(Mean of Square): {Signal_power}')
    return Signal_power


# Feature 9: Signal Energy, calcultated from signal directly, equivalent to Sum of Square 
# In theory, the value calculated should be almost equal to the energy derived from FFT (Feature 5), by Parseval's theorem
def Signal_Energy(signal_series):
    # Exclude NaN values from the signal series
    signal_array = signal_series.to_numpy()
    signal_array = signal_array[~np.isnan(signal_array)]
    signal_array_abs = np.abs(signal_array)
    signal_array_abs_square = np.square(signal_array_abs)
    signal_energy = signal_array_abs_square.sum()
    #print(f'Signal_energy(Sum of Square): {signal_energy}')
    return signal_energy


# Feature 10: Difference between Maximum and Minimum
def Min_max_difference(signal_series):
    minimum = signal_series.min()
    maximum = signal_series.max()
    Min_max_difference = maximum - minimum
    #print(f'Min_max_difference: {Min_max_difference}')
    return Min_max_difference


# Feature 11: Zero crossing (with threshold that can be chosen.
def Zero_crossing(signal_series, zc_threshold = 0):
    time_start = time.time()
    adjacent_multiplication_list = []
    signal_array = signal_series.to_numpy()
    for i in range(len(signal_array)-1):
        adjacent_multiplication_list.append(signal_array[i]*signal_array[i+1])
        
    zero_crossing = 0
    for i in adjacent_multiplication_list:
        if i < -zc_threshold:
            zero_crossing += 1
    #print(f'num of zero_crossing: {zero_crossing}')
    # print("Time taken: {} seconds".format(time.time()-time_start)) 
    return zero_crossing

'''
 See https://www.frontiersin.org/articles/10.3389/fnins.2021.667907/full  for the next few features
'''


# Feature 12: Slope sign changes (SSC): calculate the number of time that slope changes sign
def Slope_sign_changes(signal_series, ssc_threshold = 0):
    signal_series_diff = signal_series.diff()
    Slope_sign_changes = Zero_crossing(signal_series_diff,zc_threshold=ssc_threshold)
    #print(f'num of Slope_sign_changes: {Slope_sign_changes}')


# Feature 13: Wilson Amplitude: The number of times the change in the signal amplitudes of two consecutive samples exceeds the standard deviation.
def Wilson_Amplitude(signal_series):
    signal_series_diff_abs = signal_series.diff().abs()
    StD = signal_series.std()
    
    WA =  0
    for i in signal_series_diff_abs:
        if i > StD:
            WA += 1
    #print(f'Wilson Amplitude: {WA}')
    return WA


# Feature 14: Root Mean Square (RMS): The root of mean of square or v-order 2.
def Root_Mean_Square(signal_series):
    # Exclude NaN values from the signal series
    mean_square = Signal_Power(signal_series)
    RMS = np.sqrt(mean_square)
    #print(f'Root mean square: {RMS}')
    return RMS


# Feature 15: Cubic mean or V-order 3 (V3).
def V3(signal_series):
    signal_array = signal_series.to_numpy()
    # Exclude NaN values from the signal array
    signal_array = signal_array[~np.isnan(signal_array)]
    mean_of_cube = np.power(signal_array, 3).mean()
    V3 = np.cbrt(mean_of_cube)
    #print(f'Cubic mean: {V3}')
    return V3

# Feature 16: Log detector (LD): The exponential of the average of the log data. 
# (Mathematically, this is in theory the same as geometric mean) 
def Log_detector(signal_series):
    signal_series_abs = signal_series.abs()
    signal_array_abs = signal_series_abs.to_numpy()
    signal_array_abs_log = np.log(signal_array_abs)
    log_detector = exp(signal_array_abs_log.mean())
    #print(f'Log detector: {log_detector}')
    return log_detector


# Feature 17: Difference absolute standard deviation (DABS): Standard deviation of the absolute of the differential data.
# 2 mathematical equivalent (almost) implementations are provided below
def DABS(signal_series):
    DABS = signal_series.diff().std()
    #print(f'Standard deviation of the absolute of the differential data: {DABS}')
    return DABS

def DABS2(signal_series):
    signal_series_diff_square =np.square(signal_series.diff())
    DABS2 = np.sqrt(signal_series_diff_square.mean())
    #print(f'DABS2: {DABS2}')


# Feature 18: Maximum_fractal_length: Equivalent to the log of DABS PLUS an offset that is equal to (1/2)*log(N − 1)
# I suspect https://www.frontiersin.org/articles/10.3389/fnins.2021.667907/full contains typo. It should be 'Plus' the offset.
# 2 mathematical equivalent (almost) implementations are provided below
def Maximum_fractal_length(signal_series):
    DABS = signal_series.diff().std()
    Maximum_fractal_length = np.log(DABS) + 0.5 * np.log(len(signal_series))
    #print(f'Maximum_fractal_length: {Maximum_fractal_length}')
    return Maximum_fractal_length

def Maximum_fractal_length2(signal_series):
    signal_series_diff_square =np.square(signal_series.diff())
    Maximum_fractal_length2 = 0.5 * np.log(signal_series_diff_square.sum())
    #print(f'Maximum_fractal_length2: {Maximum_fractal_length2}')
    return Maximum_fractal_length2


# Feature 19: Myopulse percentage rate (MPR): the number of times the absolute of the data exceeds the standard deviation.
def Myopulse_percentage_rate(signal_series):
    signal_series_abs = signal_series.abs()
    StD = signal_series.std()
    MPR =  0
    for i in signal_series_abs:
        if i > StD:
            MPR += 1
    #print(f'Myopulse_percentage_rate: {MPR}')
    return MPR


# Plotting PSD
def Plot_PSD_welch(signal_series):
    plt.close()
    f, Pxx_den = signal.welch(signal_series,fs=30000,nperseg=512)
    plt.semilogx(f, Pxx_den)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    #print(len(Pxx_den))
    plt.show()


# Feature 20: Mean Frequency: calculated based on the power sepctral density. 
def Mean_Frequency(signal_series):
    f, Pxx_den = signal.welch(signal_series,fs=30000,nperseg=512)
    # Check if Pxx_den is all zeros to avoid division by zero
    sum_of_power= Pxx_den.sum()   # the normalization factor
    dot_product = np.dot(f, Pxx_den) # frequency weighted by power
    mean_freq = dot_product/sum_of_power
    if np.isnan(mean_freq):
        mean_freq = 0
    #print(f'mean frequency:{mean_freq}Hz')
    return mean_freq

def Feature_extraction_1(signal_df,features_list,channels_list, verbose=False):
    '''   
        Function that executes all the feature extractions and save the results in a pandas dataframe

        Parameters
        ------------
        signal_df: all the signal in the form of dataframe 
        features_list: the list of features wanted to extract
        channels_list: the list of channels that we want for channel extraction. 
                        Use signal_df.columns[:x] for convenience

        Return
        --------------
        features_table: the data stored in a dataframe
    '''
   
    features_list_string = ['channel'] + features_list
    features_table = pd.DataFrame(columns=features_list_string)
    for ch in channels_list:
        #print(ch)
        result_list = []
        result_list.append(str(ch))
        #print(result_list)
        for function_name in features_list:
            #print(function_name)
            # Use globals() to get the function by name
            if function_name  in globals() and callable(globals()[function_name]):
                method = globals()[function_name]
            result_list.append(method(signal_df['%s'%ch]))
        features_table.loc[len(features_table)] = result_list
    features_table.set_index('channel')
    return features_table


if __name__ == '__main__':

    """ EXAMPLE SO_CFAR
    num_std = 1         # number of standard deviations for the threshold
    wdur = 3  # SO-CFAR window duration in seconds
    gdur = 1    # SO-CFAR guard duration in seconds

    w = 3
    if (w % 2) == 0:
        w = w + 1
    g = 1
    if (g % 2) == 0:
        g = g + 1
    
    sig = [1, 2, 5, 6, 5, 7, 3, 4, 8, 3, 5, 2, 4, 7]   
    df_sig = pd.DataFrame({'sig':sig}) 
    indspos, indsneg, thresh = so_cfar_Zanos(df_sig, w, g, nstd= [1, 1])
    """

    """EXAMPLE NLEO"""
    # Load dataframes
    #path = ('../../datasets/feinstein/cwt_reference.mat') # feinstein.mat
    #data = sp.io.loadmat(path)
    #neural_wvl = data['sig'].reshape(-1)
    path = ('../../datasets/feinstein/IL1RKO_TNF--IL1B_4.27.2016_01__35-38min.mat') # feinstein.mat

    start = 0
    dur = 6000000 #20000000

    # Load dataframes
    neural, fs = load_matfiles(path, start=0, stop=start + dur)

    # Set neural signal to uV
    neural_sig = neural.neural_b.to_numpy() * 20  # To convert to mV in plexon

    """Check levels of reconstruction using DWT"""
    # check_MRA_wavelet_decomp(neural_sig, fs,  mra_type='dwt', wvl='db3', st=1, l=8)

    #test_compare_nleos_and_NDO(neural_sig)
    #linear_envelope(neural_sig, sampling_rate=fs, freqs=[10, 400], lfreq=4, DBplot=True)
    
        
    
    DWT_reconstructed_denoise = DWT_with_denoising(neural_sig, fs,thres_type="hard", DBplot=False,
                                    wavelet='db3', level=8,
                                    start_level=1, end_level=5)
    DWT_reconstructed = wavelet_decomp(neural_sig, fs, type='dwt', wavelet='db3', level=8,
                                    start_level=1, end_level=5)
    """
    SWT_reconstructed_denoise = SWT_with_denoising(neural_sig, fs, thres_type="hard", DBplot=False,
                                                   wavelet='db3', level=1,
                                                   start_level=5)

    SWT_reconstructed = wavelet_decomp(neural_sig, fs, type='swt',
                                                   wavelet='db3', level=1,
                                                   start_level=5)

    plot_signals_together(neural_sig, DWT_reconstructed, DWT_reconstructed_denoise)#,
    plot_signals_together(neural_sig, SWT_reconstructed, SWT_reconstructed_denoise)


    plt.show()
    """

    """ Test on SWT decomposition
     # Run check_MRA_wavelet_decomp() as below.
     # Best decomposition starting at 5 and making 1 step (level=1) 
     # However this captures the ECG components and not so well the neural spikes
     # Conclusion: SWT not a good option
    """
    """Check levels of reconstruction using SWT"""
    # check_MRA_wavelet_decomp(neural_sig, fs, mra_type='swt', wvl='db3', st=0, l=6)

    plt.show()

    
 # --------Feature extraction-------- Wilson

      