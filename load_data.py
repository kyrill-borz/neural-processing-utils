# Libraries
import numpy as np
import pandas as pd
import os
import sys
import scipy.io
import datetime
import matplotlib.pyplot as plt
from scipy import signal
import glob
import dask.dataframe as dd
import dask.array as da
import polars as pl
from pathlib import Path
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfile

from utils.load_intan_rhs_format.load_intan_rhs_format import read_rhs_data
#sys.path.append('C:/Users/amparo.guemesg/Documents/pyneural/pyNeural/datasets')
sys.path.append('../datasets')


# -------------------------------------------------------------------
#                           constants
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#                            helpers
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#                           EMPTY VECTORS
# -------------------------------------------------------------------


def empty_glucose(fs, nsamples):
    """This methods creates an empty glucose dataframe.

    .. note: The glucose dataframe has two columns time and glucose.
    .. note: Empty glucose values are represented by NaN.

    Parameters
    ----------
    fs:

    nsamples:

    Returns
    -------
    """
    # Create time vector
    time = np.arange(nsamples) / fs
    # Create glucose vector
    glucose = np.full(time.shape, np.nan)
    # Return dataframe
    return pd.DataFrame({'seconds': time, 'glucose': glucose})


def empty_hormones(fs, nsamples):
    """This methods creates an empty glucose dataframe.

    .. note: The glucose dataframe has two columns time and glucose.
    .. note: Empty glucose values are represented by NaN.

    Parameters
    ----------
    fs:

    nsamples:

    Returns
    -------
    """
    # Create time vector
    time = np.arange(nsamples) / fs
    # Create glucose vector
    insulin = np.full(time.shape, np.nan)
    glucagon = np.full(time.shape, np.nan)
    # Return dataframe
    return pd.DataFrame({'seconds': time, 'insulin': insulin, 'glucagon': glucagon})


def empty_bsamples(fs, nsamples):
    """This methods creates an empty blood samples dataframe.

    .. note: The glucose dataframe has two columns time and glucose.
    .. note: Empty glucose values are represented by NaN.

    Parameters
    ----------
    fs:

    nsamples:

    Returns
    -------
    """
    # Create time vector
    time = np.arange(nsamples) / fs

    # Create glucose, insulin and glucagon vector
    glucose = np.full(time.shape, np.nan)
    insulin = np.full(time.shape, np.nan)
    glucagon = np.full(time.shape, np.nan)

    # Return dataframe
    return pd.DataFrame({'seconds': time, 'glucose': glucose,
                         'insulin': insulin, 'glucagon': glucagon})


def empty_stimuli(fs, nsamples):
    """This method creates and empty stimuli dataframe.

    Parameters
    ----------

    Returns
    -------
    """
    # Create time vector
    time = np.arange(nsamples) / fs
    # Create amplitude/duration vectors
    amplitude = np.zeros(time.shape)
    duration = np.zeros(time.shape)
    # Return dataframe
    return pd.DataFrame({'seconds': time,
                         'amplitude': amplitude,
                         'duration': duration})


# -------------------------------
# Generate stimuli from dataframe
# -------------------------------
def generate_stimuli(fs, nsamples, setup):
    """This method generates the stimuli signal.

    .. note: Would it be useful to create a function time2idx that 
             allows to input the time unit in order to compute the
             array index?

    Parameters
    ----------
    fs: float-like (hz)
      The sample frequency

    duration: int-like (s)
      The duration of the signal in seconds

    setup: dataframe-like
      The stimuli setup which is a dataframe with the columns pulse amplitude
      (amplitude), pulse duration (duration), and the time interval in which 
      it is applied in seconds (start, end).

    Returns
    -------
    signal_amplitude: np.array
      The signal with the amplitudes of the stimuli

    signal_duration: np.array
      The signal with the duration of the stimule
    """
    # Create pulse amplitude/duration signals
    amplitude = np.zeros(nsamples)
    duration = np.zeros(nsamples)
    frequency = np.zeros(nsamples)
    time = np.arange(nsamples) / fs
    print('Fs in stimuli: %s' % fs)

    # Fill the signals
    for index, row in setup.iterrows():
        # Get indexes
        idxs = np.arange(row.start * fs, row.end * fs).astype(np.int32)

        # Fill signal
        np.put(amplitude, idxs, row.amplitude)
        np.put(duration, idxs, row.duration)
        np.put(frequency, idxs, row.frequency)

    # Create dataframe
    stimuli = pd.DataFrame({'seconds': time,
                            'amplitude': amplitude,
                            'duration': duration,
                            'frequency': frequency})

    # Return
    return stimuli

# -------------------------------------------------------------------
#                           LOAD METHODS: SIMPLE
# -------------------------------------------------------------------
# ----------------
# Load methods
# ----------------


def load_setup(path, verbose=1):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ---------
    # Paths
    # ---------
    path_ecg = '%s/_ecg.csv' % path
    path_glucose = '%s/_bsamples.csv' % path
    path_stimuli = '%s/_stimuli.csv' % path
    path_hormones = '%s/_hormones.csv' % path

    # ---------
    # Load ecg
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading ECG from: %s" % path_ecg)

    # Load ecg
    ecg = pd.read_csv(path_ecg)

    # Frequency sample
    fs = 1 / (ecg.seconds.iloc[1] - ecg.seconds.iloc[0])

    # Number of samples
    n = len(ecg)

    # ------------------
    # Default dataframes
    # ------------------
    glucose = empty_glucose(fs=fs, nsamples=n)
    stimuli = empty_stimuli(fs=fs, nsamples=n)
    hormones = empty_hormones(fs=fs, nsamples=n)

    # --------------
    # Load glucose
    # --------------
    # Load glucose
    if os.path.isfile(path_glucose):
        glucose = load_glucose(path=path_glucose, fs=fs, nsamples=n)

    # --------------
    # Load hormones
    # --------------
    # Load insulin and glucagon
    if os.path.isfile(path_hormones):
        hormones = load_hormones(path=path_hormones, fs=fs, nsamples=n)

    # --------------------
    # Load stimuli
    # --------------------
    if os.path.isfile(path_stimuli):
        stimuli = load_stimuli(path=path_stimuli, fs=fs, nsamples=n)

    # To avoid that timeframe starts in 0
    # ecg.seconds.iloc[0] = 0.00001

    # Return
    return ecg, glucose, hormones, stimuli, fs, '.csv'


def load_ecg(path, verbose=1):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """

    # ---------
    # Load ecg
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading ECG from: %s" % path)

    # Load ecg
    ecg = pd.read_csv(path)

    # To avoid that timeframe starts in 0
    ecg.seconds.iloc[0] = 0.00001

    # Frequency sample
    fs = 1 / (ecg.seconds.iloc[1] - ecg.seconds.iloc[0])

    # Return
    return ecg, fs, '.csv'


def load_stimuli(path, fs=None, nsamples=None, verbose=1):
    """This method loads the stimuli.

    .. note:

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading stimuli from: %s" % path)

    # Load stimuli setup
    stimuli_setup = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return stimuli_setup

    # Create signal
    stimuli = generate_stimuli(fs=fs,
                               nsamples=nsamples,
                               setup=stimuli_setup)

    # Return
    return stimuli

def load_bsamples_start_end(path, fs=None, start=0, end=0, nsamples=None, verbose=1):
    """This method loads the whole glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading blood samples from: %s" % path)

    # Load blood samples
    bsamples = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return bsamples
    #print(fs)
    #print(nsamples)

    # Create time vector: calculates the time values corresponding to each sample in a recording, with the first sample mapped to the start time and the last sample mapped to the end time.
    ##  np.arange(nsamples) generates an array of integers from 0 to nsamples - 1. This array represents the sample indices.
    ## (end - start) calculates the duration of the recording.
    ## (np.arange(nsamples) / nsamples * (end - start)) calculates the time elapsed from the start time to each sample index. It multiplies the proportion of the duration corresponding to each sample by the total duration.
        # Convert the start timestamp to a Unix timestamp (seconds since the Unix epoch)
    start_unix_timestamp = start.timestamp()
    time = (np.arange(nsamples) / nsamples * (end - start).total_seconds()) + start_unix_timestamp # AG 16/03/2024 Added start_unix_timestamp when loading chronic VN, didn't work for periods of recordings (not full traces)
    
    print(time)
    print(start)
    print(end.timestamp)
    #print(start.hour*3600 +start.minute*60 + start.second)
    #time = time+start.hour*3600 +start.minute*60 + start.second
    print(time[-1])
    print(len(time))
    print(bsamples.seconds_all.values)
    idx = []
    '''
    # Find idxs where blood samples times should be inserted
    for sec in bsamples.seconds_all.values:
        if (len(np.where(sec==time)[0])==0):
            print('Discarding sec: %s'%sec)
            continue
        else:
            print('Appending sec: %s'%sec)
            idx.append(np.where(time==sec)[0][0])
    #idx = np.searchsorted(time, bsamples.seconds_all.values)
    '''
    # Find indices of closest values in time array
    idx = np.searchsorted(time, bsamples.seconds_all.values)  # return more idx than expected then time doesn't start with a value contained in bsamples.
    #idx = np.where(np.isin(time, bsamples.seconds_all))[0]
    #idx = [np.abs(time - val).argmin() for val in bsamples.seconds_all]

    # Clip indices to be within the bounds of the time array
    idx = np.clip(idx, 0, len(time)-1)

    print('idx: %s' %idx)
    
    unique_idx = np.unique(idx)
    print('unique_idx : %s' %unique_idx)
    #unique_idx[-1] = unique_idx[-1]-1

    #Filter the dataframe from start to nsamples
 
    #bsamples2 = bsamples.iloc[np.where(bsamples.seconds_all.values>=int(time[0]))[0][0]:(np.where(bsamples.seconds_all.values<=int(time[-1]))[0][-1]+1)]
    bsamples2 = bsamples

    # Create glucose vector
    vglucose = np.zeros(len(time))
    vglucose[:] = np.nan
    
    '''
    if len(unique_idx)!=len(bsamples2.glucose):
        unique_idx = np.unique(idx)[:-1] # For some reason, sometimes it takes more than needed
        vglucose[unique_idx] = bsamples2.glucose
    else:
    '''
    # Ojo solo sirve para los primeros valores del array, si no borrar directamente filas del excel
    # vglucose[unique_idx] = bsamples2.glucose[0:len(unique_idx)] #AG 16/03 Created code below to load only corresponding rows

    # Assuming start and end are timestamps indicating the start and end times
    bsamples2 = bsamples2[(bsamples2["seconds_all"] >= start.timestamp()) & (bsamples2["seconds_all"] <= end.timestamp()+1)]
    print(bsamples2["glucose"].values)

    # Then, assign the glucose values from bsamples2 to vglucose
    vglucose[unique_idx] = bsamples2["glucose"].values

    # Create insulin vector
    vinsulin = np.zeros(time.shape)
    vinsulin[:] = np.nan
    vinsulin[unique_idx] = bsamples2.insulin.values

    # Create glucagon vector
    vglucagon = np.zeros(time.shape)
    vglucagon[:] = np.nan
    vglucagon[unique_idx] = bsamples2.glucagon.values

    # Create gluc_change vector
    vgluc_ch = np.zeros(time.shape)
    vgluc_ch[:] = np.nan
    vgluc_ch[unique_idx] = bsamples2.change_label.values

    # Return
    return pd.DataFrame({'seconds': time, 'glucose': vglucose, #+bsamples.seconds.values[0]
                         'insulin': vinsulin, 'glucagon': vglucagon, 'change_label': vgluc_ch})


def load_bsamples(path, fs=None, start=0, nsamples=None, verbose=1):
    """This method loads the whole glucose..

    updated with code above, this one doesn't seem to work well 

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading blood samples from: %s" % path)
    # Load blood samples
    bsamples = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return bsamples
    #print(fs)
    #print(nsamples)

    # Create time vector every 1 min from length glucose data
    time_minutes = np.arange(0, nsamples+fs, fs*60) / fs
    print(time_minutes)
    print(start)
    time = time_minutes+start
    print(time_minutes)
    print(time_minutes[-1])
    print(len(time))
    print(bsamples.seconds_all.values)
    idx = []
    '''
    # Find idxs where blood samples times should be inserted
    for sec in bsamples.seconds_all.values:
        if (len(np.where(sec==time)[0])==0):
            print('Discarding sec: %s'%sec)
            continue
        else:
            print('Appending sec: %s'%sec)
            idx.append(np.where(time==sec)[0][0])
    #idx = np.searchsorted(time, bsamples.seconds_all.values)
    '''
    # Find indices of closest values in time array
    idx = np.searchsorted(time_minutes, bsamples.seconds_all.values)

    # Clip indices to be within the bounds of the time array
    idx = np.clip(idx, 0, len(time_minutes)-1)

    print(idx)
    print(bsamples)
    
    unique_idx = np.unique(idx)
    print(unique_idx)
    #unique_idx[-1] = unique_idx[-1]-1

    #Filter the dataframe from start to nsamples
    #bsamples2 = bsamples.iloc[np.where(bsamples.seconds_all.values>=int(time[0]))[0][0]:(np.where(bsamples.seconds_all.values>=int(time[-1]))[0][0]+1)]
    bsamples2 = bsamples.iloc[np.where(bsamples.seconds_all.values>=int(time[0]))[0][0]:(np.where(bsamples.seconds_all.values<=int(time[-1]))[0][-1]+1)]
    
    #bsamples2 = bsamples[bsamples.seconds_all.values<=int(time[-1])]
    #print(bsamples2)

    # Create glucose vector
    vglucose = np.zeros(len(time))
    vglucose[:] = np.nan
    
    '''
    if len(unique_idx)!=len(bsamples2.glucose):
        unique_idx = np.unique(idx)[:-1] # For some reason, sometimes it takes more than needed
        vglucose[unique_idx] = bsamples2.glucose
    else:
    '''
    vglucose[unique_idx] = bsamples2.glucose

    # Create insulin vector
    vinsulin = np.zeros(time.shape)
    vinsulin[:] = np.nan
    vinsulin[unique_idx] = bsamples2.insulin

    # Create glucagon vector
    vglucagon = np.zeros(time.shape)
    vglucagon[:] = np.nan
    vglucagon[unique_idx] = bsamples2.glucagon

    # Create gluc_change vector
    vgluc_ch = np.zeros(time.shape)
    vgluc_ch[:] = np.nan
    vgluc_ch[unique_idx] = bsamples2.change_label

    # Return
    return pd.DataFrame({'seconds': time, 'glucose': vglucose, #+bsamples.seconds.values[0]
                         'insulin': vinsulin, 'glucagon': vglucagon, 'change_label': vgluc_ch})



def load_glucose(path, fs=None, nsamples=None, verbose=1):
    """This method loads the glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading glucose from: %s" % path)

    # Load glucose
    glucose = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return glucose

    # Create time vector
    time = np.arange(nsamples) / fs

    # Find idxs where glucose times should be inserted
    idx = np.searchsorted(time, glucose.seconds.values) - 1

    # Create glucose vector (mean value)
    vglucose = np.zeros(time.shape)
    vglucose[:] = np.nan
    vglucose[idx] = glucose.glucose

    # Create glucose std vector
    glucstd = np.zeros(time.shape)
    glucstd[:] = np.nan
    glucstd[idx] = glucose.glucstd

    # Return
    return pd.DataFrame({'seconds': time, 'glucose': vglucose, 'glucstd': glucstd})


def load_manualHR(path, fs=None, nsamples=None, verbose=1):
    """This method loads the glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading manual HR from: %s" % path)

    # Load glucose
    manual_hr = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return manual_hr

    # Create time vector
    time = np.arange(nsamples) / fs

    # Find idxs where glucose times should be inserted
    idx = np.searchsorted(time, manual_hr.seconds.values) - 1

    # Create glucose vector
    vmanual_hr = np.zeros(time.shape)
    vmanual_hr[:] = np.nan
    vmanual_hr[idx] = manual_hr.HR

    # Return
    return pd.DataFrame({'seconds': time, 'manualHR': vmanual_hr})


def load_hormones(path, fs=None, nsamples=None, verbose=1):
    """This method loads the glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading insulin & glucagon from: %s" % path)

    # Load glucose
    hormones = pd.read_csv(path)

    # Return file
    if fs is None or nsamples is None:
        return hormones

    # Create time vector
    time = np.arange(nsamples) / fs

    # Find idxs where glucose times should be inserted
    idx = np.searchsorted(time, hormones.seconds.values) - 1

    # Create insulin vector
    vins = np.zeros(time.shape)
    vins[:] = np.nan
    vins[idx] = hormones.insulin

    # Create glucagon vector
    vgluc = np.zeros(time.shape)
    vgluc[:] = np.nan
    vgluc[idx] = hormones.glucagon

    # Return
    return pd.DataFrame({'seconds': time, 'insulin': vins, 'glucagon': vgluc})


def load_processed(path, verbose=1, deltaTime=False):
    """This method loads...

    .. note:
    .. note:

    Parameters
    ----------

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading processed from: %s" % path)

    # Load processed
    processed = pd.read_csv(path, parse_dates=['time'], index_col='time')  # 

    # Sample frequency
    fs = 1 / (processed.seconds.iloc[1] - processed.seconds.iloc[0])

    if deltaTime:
        # Set delta time index
        processed.index = pd.TimedeltaIndex(processed.seconds, unit='s')

    # Return
    return processed, fs

def load_hemod(path, verbose=1, resample=939):
    """This method loads exclusively HR (from processed ecg) and blood pressure

    .. note:
    .. note:

    Parameters
    ----------
    path: path for processed ecg
    resample: sampling frequency to resample if it runs out of memory

    Returns
    -------
    data: dataframe with time (seconds), Hr (bpm) and BP (bp)
    resample: sampling frequency in the new dataframe

    """
    # Show information
    if (verbose > 0):
        print("Loading processed from: %s" % path)

    # Save folder where data is being loaded
    folder = os.path.dirname(path)

    # Load BP
    path_bp = '%s/_bp.h5' % folder

    # Load HR dataframes
    ecg = pd.read_csv(path, usecols=['time', 'seconds', 'bpm'],
                      parse_dates=['time'], index_col='time')
    ecg = ecg.interpolate(limit_direction='backward', axis=0)

    # Duration experiment
    dur = int(ecg.seconds.iloc[-1] / 60)

    # Load blood pressure from raw signal
    bp, fs_bp = load_bp_chunk(path_bp, start=0, stop=dur)

    start = int(ecg.seconds.iloc[0])
    stop = dur
    n_samples = (stop - start) * 60 * resample

    data = pd.DataFrame({'seconds': np.linspace(start * 60, stop * 60, n_samples),
                         'bpm': signal.resample(ecg['bpm'], n_samples),
                         'bp': signal.resample(bp['bp_signal'], n_samples)})
    data.seconds.iloc[0] = 0.0001

    # Return
    return data, resample


def load_sampletime(path, fs=None, nsamples=None, verbose=1):
    """This method loads the whole glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading blood samples from: %s" % path)

    # Load blood samples
    bsamples = pd.read_csv(path)

    # Create time vector from length glucose data
    time = np.arange(nsamples) / fs

    # Return file
    if fs is None or nsamples is None:
        return bsamples

    # Find idxs where tail samples times should be inserted
    seconds_tail = bsamples.seconds.iloc[bsamples['location'].values == 'tail']
    idx_tail = np.searchsorted(time, seconds_tail.values) - 1
    # print(idx_tail)

    # Create tail vector
    vtail = np.zeros(time.shape)
    vtail[idx_tail] = 1

    # Find idxs where tube samples times should be inserted
    seconds_tube = bsamples.seconds.iloc[bsamples['location'].values == 'tube']
    idx_tube = np.searchsorted(time, seconds_tube.values) - 1

    # Create tube vector
    vtube = np.zeros(time.shape)
    vtube[idx_tube] = 1

    # Find idxs where artery samples times should be inserted
    seconds_artery = bsamples.seconds.iloc[bsamples['location'].values == 'artery']
    idx_artery = np.searchsorted(time, seconds_artery.values) - 1
    # print(idx_artery)

    # Create artery vector
    vartery = np.zeros(time.shape)
    vartery[idx_artery] = 1

    # Find idxs where clots times should be inserted
    seconds_clot = bsamples.seconds.iloc[bsamples['location'].values == 'clot']
    idx_clot = np.searchsorted(time, seconds_clot.values) - 1
    # print(idx_artery)

    # Create clots vector
    vclot = np.zeros(time.shape)
    vclot[idx_clot] = 1

    # Find idxs where comments times should be inserted
    seconds_comment = bsamples.seconds.iloc[bsamples['location'].values == 'comment']
    idx_comment = np.searchsorted(time, seconds_comment.values) - 1
    # print(idx_artery)

    # Create comments vector
    vcomment = np.zeros(time.shape)
    vcomment[idx_comment] = 1

    # Return
    return pd.DataFrame({'seconds': time, 'clot': vclot, 'tail': vtail,
                         'tube': vtube, 'artery': vartery,
                         'comment': vcomment})

# -------------------------------------------------------------------
#                           LOAD METHODS: CHUNKS
# -------------------------------------------------------------------


def load_setup_chunks(path, resample=300, start=0, stop=20, fs=939, verbose=1):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup
    start : min
    stop : min
    fs : HR sampling frequency

    Returns
    -------

    """
    # ---------
    # Paths
    # ---------
    path_ecg = '%s/_ecg.h5' % path
    # path_glucose = '%s/_glucose.csv' % path
    path_bsamples = '%s/_bsamples.csv' % path
    path_stimuli = '%s/_stimuli.csv' % path
    # path_hormones = '%s/_hormones.csv' % path

    # ---------
    # Load ecg
    # ---------
    sample_start = start * 60 * fs
    sample_stop = stop * 60 * fs

    # Load ecg
    ecg, fs, ext = load_ecg_chunk(
        path_ecg, start=sample_start, stop=sample_stop)
    n = len(ecg)

    if resample:
        fs = resample
        n = (stop - start) * 60 * resample
        sample_start = start * 60 * fs
        sample_stop = stop * 60 * fs
        print(sample_start)
        print(sample_stop)
        ecg = pd.DataFrame({'seconds': np.linspace(start * 60, stop * 60, n),
                            'ecg_signal': signal.resample(ecg.ecg_signal, n)})
        # To avoid that timeframe starts in 0
        ecg.seconds.iloc[0] = 0.00001

        # print(ecg)
        # plt.plot(ecg.seconds, ecg.ecg_signal)
        # plt.show()

    # ------------------
    # Default dataframes
    # ------------------
    # glucose = empty_glucose(fs=fs, nsamples=n)
    # hormones = empty_hormones(fs=fs, nsamples=n)
    stimuli = empty_stimuli(fs=fs, nsamples=n)
    bsamples = empty_bsamples(fs=fs, nsamples=n)

    """
    # --------------
    # Load glucose
    # --------------
    # Load glucose
    if os.path.isfile(path_glucose):
        glucose, nsamples = load_gluose_chunks(path=path_glucose, fs=fs)
        glucose = pd.DataFrame({'seconds': glucose.seconds.iloc[sample_start:sample_stop].values,
                                'glucose': glucose.glucose.iloc[sample_start:sample_stop].values})

    # --------------
    # Load hormones
    # --------------
    # Load insulin and glucagon
    if os.path.isfile(path_hormones):
        hormones = load_hormones(path=path_hormones, fs=fs, nsamples=nsamples)
        hormones = pd.DataFrame({'seconds': hormones.seconds.iloc[sample_start:sample_stop].values,
                                'insulin': hormones.insulin.iloc[sample_start:sample_stop].values,
                                'glucagon': hormones.glucagon.iloc[sample_start:sample_stop].values})   
    """
    # -------------------
    # Load blood samples
    # -------------------
    # Load glucose, insulin and glucagon
    if os.path.isfile(path_bsamples):
        bsamples, nsamples = load_bsamples_chunks(path=path_bsamples,
                                                  fs=fs, n=n)  # nsamplamples = n
        bsamples = pd.DataFrame({'seconds': bsamples.seconds.iloc[sample_start:sample_stop].values,
                                 'glucose': bsamples.glucose.iloc[sample_start:sample_stop].values,
                                 'insulin': bsamples.insulin.iloc[sample_start:sample_stop].values,
                                 'glucagon': bsamples.glucagon.iloc[sample_start:sample_stop].values,
                                 'glucstd': bsamples.glucstd.iloc[sample_start:sample_stop].values})

    # --------------------
    # Load stimuli
    # --------------------
    if os.path.isfile(path_stimuli):
        stimuli = load_stimuli(path=path_stimuli, fs=fs,
                               nsamples=n)  # nsamples
        stimuli = pd.DataFrame({'seconds': stimuli.seconds.iloc[sample_start:sample_stop].values,
                                'amplitude': stimuli.amplitude.iloc[sample_start:sample_stop].values,
                                'duration': stimuli.duration.iloc[sample_start:sample_stop].values,
                                'frequency': stimuli.frequency.iloc[sample_start:sample_stop].values})

    # Return
    return ecg, bsamples, stimuli, fs, ext


def load_ecg_chunk(path, verbose=1, start=0, stop=200000, fs=939):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ---------
    # Load ecg
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading HR from: %s" % path)

    # Load ecg
    sample_start = start * 60 * fs
    sample_stop = stop * 60 * fs

    store = pd.HDFStore(path)
    ecg = store.select('data', start=sample_start,
                       stop=sample_stop,
                       columns=['seconds', 'ecg_signal'])

    # Close store
    store.close()

    # Frequency sample
    fs = 1 / (ecg.seconds.iloc[-1] - ecg.seconds.iloc[-2])

    # Return
    return ecg, fs, '.h5'


def load_bp_chunk(path, verbose=1, start=0, stop=200000, fs=191):
    """This method loads...

    .. note: It is assumed that bp is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ---------
    # Load bp
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading BP from: %s" % path)

    # Load ecg
    sample_start = start * 60 * fs
    sample_stop = stop * 60 * fs

    store = pd.HDFStore(path)
    bp = store.select('data', start=sample_start,
                      stop=sample_stop,
                      columns=['seconds', 'bp_signal'])

    # Close store
    store.close()

    # Frequency sample
    fs = 1 / (bp.seconds.iloc[-1] - bp.seconds.iloc[-2])

    # Return
    return bp, fs


def load_glucose_chunks(path, fs=None, verbose=1):
    """This method loads the whole glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading glucose from: %s" % path)

    # Load glucose
    glucose = pd.read_csv(path)

    # Create time vector from length glucose data
    nsamples = int(glucose.seconds.values[-1] * fs)
    time = np.arange(nsamples) / fs

    # Return file
    if fs is None:
        return glucose

    # Create time vector
    # time = np.arange(nsamples) / fs
    # print(time)

    # Find idxs where glucose times should be inserted
    idx = np.searchsorted(time, glucose.seconds.values) - 1

    # Create glucose vector
    vglucose = np.zeros(time.shape)
    vglucose[:] = np.nan
    vglucose[idx] = glucose.glucose

    # Create glucse std vector
    vglucstd = np.zeros(time.shape)
    vglucstd[:] = np.nan
    vglucstd[idx] = glucose.glucstd

    # Return
    return pd.DataFrame({'seconds': time, 'glucose': vglucose,
                         'glucstd': vglucstd}), nsamples


def load_bsamples_chunks(path, fs=None, verbose=1, n=False):
    """This method loads the whole glucose..

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading blood samples from: %s" % path)

    # Load glucose
    bsamples = pd.read_csv(path)

    # Create time vector from length glucose data
    nsamples = int(bsamples.seconds.values[-1] * fs)

    if n:
        nsamples = n

    time = np.arange(nsamples) / fs

    # Return file
    if fs is None:
        return bsamples

    # Find idxs where blood samples times should be inserted
    idx = np.searchsorted(time, bsamples.seconds.values) - 1

    # Create glucose vector
    vglucose = np.zeros(time.shape)
    vglucose[:] = np.nan
    vglucose[idx] = bsamples.glucose

    # Create glucse std vector
    vglucstd = np.zeros(time.shape)
    vglucstd[:] = np.nan
    vglucstd[idx] = bsamples.glucstd

    # Create insulin vector
    vinsulin = np.zeros(time.shape)
    vinsulin[:] = np.nan
    vinsulin[idx] = bsamples.insulin

    # Create glucagon vector
    vglucagon = np.zeros(time.shape)
    vglucagon[:] = np.nan
    vglucagon[idx] = bsamples.glucagon

    # Return
    return pd.DataFrame({'seconds': time, 'glucose': vglucose,
                         'insulin': vinsulin, 'glucagon': vglucagon,
                         'glucstd': vglucstd}), nsamples


def load_stimuli_chunks(path, fs=None, verbose=1):
    """This method loads the stimuli. NOT USED

    .. note:

    Parameters
    ----------
    path :

    fs :

    nsamples :

    Returns
    -------
    """
    # Show information
    if (verbose > 0):
        print("Loading stimuli from: %s" % path)

    # Load stimuli setup
    stimuli_setup = pd.read_csv(path)

    # Return file
    if fs is None:
        return stimuli_setup

    # Number ofsamples
    nsamples = stimuli_setup.seconds.values[-1] * fs

    # Create signal
    stimuli = generate_stimuli(fs=fs,
                               nsamples=nsamples,
                               setup=stimuli_setup)

    # Return
    return stimuli


# -------------------------------------------------------------------
#                           LOAD NEURAL
# -------------------------------------------------------------------
def load_neural_chunk(path, verbose=1, start=0, stop=200000):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ---------
    # Paths
    # ---------
    path = '%s/_neural.h5' % path

    # ---------
    # Load ecg
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading ECG from: %s" % path)

    # Load ecg

    store = pd.HDFStore(path)
    neural = store.select('data',
                          start=start,
                          stop=stop,
                          columns=['seconds', 'neural_b', 'neural_a'])
    # Close store
    store.close()

    # Frequency sample
    fs = 1 / (neural.seconds.iloc[-1] - neural.seconds.iloc[-2])

    # Return
    return neural, fs


def load_neural(path):
    """This method loads...

    .. note: This probably should go into a library that could be
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ------------
    # Load neural
    # ------------
    # Load neural
    neural_signal = np.load('%s/neural.npy' % path)
    neural_time = np.load('%s/neuralTime.npy' % path)
    print('Neural files loaded')

    # Frequency sample
    fs = 1 / (neural_time[1] - neural_time[0])

    # Return
    # return pd.DataFrame({'seconds': neural_time, 'signal': neural_signal}), fs
    return pd.DataFrame({'signal': neural_signal}), fs


# -------------------------------------------------------------------
#                           LOAD MATLAB FILES
# -------------------------------------------------------------------
def load_matfiles(path, verbose=1, start=0, stop=1000):
    """This method loads...

    .. note: It is assumed that ecg is always available.

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path : string-like
      The path for the corresponding setup

    Returns
    -------

    """
    # ---------
    # Paths
    # ---------
    path = '%s/feinstein.mat' % path

    # ---------
    # Load ecg
    # ---------
    # Show information
    if (verbose > 0):
        print("Loading neurograms from: %s" % path)

    # Load data
    data = scipy.io.loadmat(path)

    # Create dataframe
    neural_b = data['neural_b'].reshape(-1)
    seconds = data['seconds'].reshape(-1)
    neural = pd.DataFrame({'seconds': seconds.iloc[start:stop],
                           'neural_b': neural_b[start:stop]})
    # Frequency sample
    fs = 1 / (neural.seconds.iloc[-1] - neural.seconds.iloc[-2])

    # Return
    return neural, fs

def extractZ(data):
    print('Extracting Impedance Values..')
    #columns = []
    channels=[]
    Zmagnitudes = []
    Zphases = []
    port_numbers = []
    for i, el in enumerate(data['amplifier_channels']):
        magnitude = el['electrode_impedance_magnitude']
        phase = el['electrode_impedance_phase']
        port_number = el['port_number']
        channel = el['native_order']
        # Append data to respective lists
        Zmagnitudes.append(magnitude)
        Zphases.append(phase)
        port_numbers.append(port_number)
        channels.append(channel)

    #columns.extend('%s_%s' %(channel, port_number))

    df_Z = pd.DataFrame({
    'port_number': port_numbers,
    'amplifier_channel': channels,
    'impedance_magnitude': Zmagnitudes,
    'impedance_phase': Zphases
    })
    return df_Z

def load_all_ports_rhs(path, fileType='rhs', downsample=1, verbose=0):
    """
    Load all .rhs files in a folder and return a dict with port-separated DataFrames and metadata.
    It also saves the processed data to files in the same folder.
    
    Returns:
        df_by_port (dict): Port-wise DataFrames with channels as columns.
        fs (float): Sampling frequency.
        channel_info (list): List of dicts with channel metadata.
        full_df (DataFrame): Complete DataFrame with all channels.
    """
    
    # Gather all files
    files = sorted(glob.glob(os.path.join(path, f'*.{fileType}'), recursive=True))
    if verbose:
        print(f"Found {len(files)} {fileType} files in: {path}")
        print(files)
    
    amp_data = []
    time_data = []
    fs = None
    all_channel_info = None

    # Loop through each file
    for count, file in enumerate(files):
        print(count)
        print(f"Loading file {count+1}/{len(files)}: {file}")
        data = read_data(file, verbose=verbose)
        
        # Sampling frequency (assume constant)
        if fs is None:
            fs = data['frequency_parameters']['amplifier_sample_rate']
        
        # Channel metadata (assume constant)
        if all_channel_info is None:
            all_channel_info = data['amplifier_channels']
        
        # Amplifier data
        amp = data['amplifier_data'].T  # shape: [samples, channels]
        amp = amp[::downsample]
        amp_data.append(amp)

        # Time vector
        t = data['t'] if 't' in data else data['t_amplifier']
        t = t[::downsample]
        time_data.append(t)

    # Concatenate across files
    amp_data = np.vstack(amp_data)  # shape: [total_samples, channels]
    time_data = np.concatenate(time_data)

    # Create column labels from amplifier_channels
    channel_names = [f"{ch['port_name']}_{ch['native_channel_name']}" for ch in all_channel_info]
    port_names = [ch['port_name'] for ch in all_channel_info]

    df_all = pd.DataFrame(amp_data, columns=channel_names)
    df_all['time'] = time_data

    # Split into port-wise DataFrames
    print('split')
    df_by_port = {}
    for port in set(port_names):
        cols = [f"{ch['port_name']}_{ch['native_channel_name']}" for ch in all_channel_info if ch['port_name'] == port]
        df_port = df_all[cols + ['time']].copy()
        df_by_port[port] = df_port

    # Save DataFrames and metadata to files in the same folder
    base_filename = os.path.basename(path.rstrip('/'))
    
    # Save full DataFrame
    full_df_filename = os.path.join(path, f"{base_filename}_full_data.csv")
    df_all.to_csv(full_df_filename, index=False)
    print(f"Saved full data to {full_df_filename}")
    
    # Save port-wise DataFrames
    for port, df_port in df_by_port.items():
        port_filename = os.path.join(path, f"{base_filename}_port_{port}_data.csv")
        df_port.to_csv(port_filename, index=False)
        print(f"Saved {port} data to {port_filename}")
    
    # Save sampling frequency and channel metadata
    metadata_filename = os.path.join(path, f"{base_filename}_metadata.pkl")
    with open(metadata_filename, 'wb') as metadata_file:
        pickle.dump({'fs': fs, 'channel_info': all_channel_info}, metadata_file)
    print(f"Saved metadata to {metadata_filename}")

    return df_by_port, fs, all_channel_info, df_all


def load_data_multich(path, start=0, dur=None, port='Port B', load_from_file=False, load_multiple_files=False, fileType='rhs', downsample=1, day='', verbose=1):
    """This method loads...

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path :      [string-like] The path for the corresponding setup
    start:      [int] sample onset
    dur  :      [int] duration in samples
    channels:   [array] channels to be selected. If a single port is used then it's always (0, 32), if two ports then port A (0,32) and port B(32, 64)
    load_from_file: [boolean] By default False loads from mat o rhs files, otherwise loads 
                    a previously stored csv or pkl with the dataframe
    downsample: [int] (default: 1) downsampling factor.
    verbose:    [int] signal to display text information (default 1 - show text)
    
    Returns
    -------
    neural:     [dataframe] Index is the time in DateTime, one column for each channel
    fs:         [float] Sampling frequency
    basename_without_ext: [string] name of the file without the extension. For storing purposes
    channels:   [array of ints] Available channels from amplifier data in rhs file

    """      

    # Extract directory of current path
    # dir_name = os.path.dirname(path)

    # Open GUI for selecting file
    Tk().withdraw()  # keep the root window from appearing
    # If data has been previously stored
    if fileType == 'rhs':
        time_key = 't'
    else:
        time_key = 't_amplifier'
        
    if load_from_file:
        if path == None:
            filepath = askopenfile(initialdir=path, title="Select previously stored data file", 
                                filetypes=[("recording", ".csv .pkl .parquet")])
            filepath = filepath.name
        else:
            filepath = path
        print('Loading from file %s' %filepath)
        channels = []

        # Load from csv: computationally expensive
        if filepath.endswith('.csv'):
            neural = pd.read_csv(filepath)
            # Check the file is a data file
            if 'time' in neural.columns:
                # Remove 'time' column and remake it as index (otherwise it's imported as String and not Dataframe)
                neural = neural.drop(columns=['time'])
                neural.index = pd.DatetimeIndex(neural.seconds * 1e9)
                neural.index.name = 'time'

                # Set time interval
                print(start)
                if dur is None:
                    stop = len(neural)
                else:
                    stop = int(start + dur)
                print('stop: %s' %stop)
                start = int(start)

                neural = neural.iloc[start:stop]

                # Get Sampling frequency
                fs = 1/(neural['seconds'].iloc[2]-neural['seconds'].iloc[1])
                print(fs)

                # Downcast it type float64 
                for col in neural.columns:
                    if col.startswith('ch_'):
                        neural[col] = neural[col].astype('float32')

                basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
                try: 
                    information = pd.read_csv('%s/%sChannel_info_%s.csv' %(path, day, basename_without_ext))
                    print(information)
                except:
                    print('information not available')
                    # Create empty dataframe with information for all channels
                    information = pd.DataFrame() #columns=['ch_string', 'intan_ch', 'Z_magnitude', 'Z_phase'])
            else:
                print('ERROR: You have selected a wrong file, try again')
                sys.exit()
        #Load from pickle: much faster
        else:
            if filepath.endswith('.pkl'):
                neural = pd.read_pickle(filepath)
                if neural.index.name == 'time':
                    # Set time interval
                    print(start)
                    if dur is None:
                        stop = len(neural)
                    else:
                        stop = int(start + dur)
                    print('stop: %s' %stop)
                    start = int(start)

                    neural = neural.iloc[start:stop]

                    # Get Sampling frequency
                    fs = 1/(neural['seconds'].iloc[2]-neural['seconds'].iloc[1])
                    print(fs)

                    # Downcast it type float64 
                    for col in neural.columns:
                        if col.startswith('ch_'):
                            neural[col] = neural[col].astype('float32')
                            channels.append(col.replace('ch_', ''))
                    print(channels)
                    basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
                    try: 
                        information = pd.read_csv('%s/%sChannel_info_%s.csv' %(path, day, basename_without_ext))
                        print(information)
                    except:
                        print('information not available')
                        # Create empty dataframe with information for all channels
                        information = pd.DataFrame() #columns=['ch_string', 'intan_ch', 'Z_magnitude', 'Z_phase'])
                else:
                    print('ERROR: You have selected a wrong file, try again')
                    sys.exit()

            if filepath.endswith('.parquet'):
                neural = pl.read_parquet(filepath)
                # neural = pd.read_parquet(filepath)
                # print('---------------')
                # print(neural)
                # print('---------------')
            # Check the file is a data file
                if 'time' in neural.columns:
                    # Set time interval
                    print(start)
                    if dur is None:
                        stop = len(neural)
                    else:
                        stop = int(start + dur)
                    print('stop: %s' %stop)
                    start = int(start)

                    neural = neural[start:stop]

                    # Get Sampling frequency
                    fs = 1/(neural['seconds'][2]-neural['seconds'][1])
                    # print("sampling freq %s" %fs)

                    # Downcast it type float64 
                    # for col in neural.columns:
                    #     if col.startswith('ch_'):
                    #         neural = neural.with_columns(
                    #             pl.col(col).cast(pl.Float32).alias(col)
                    #         )
                    #         channels.append(col.replace("ch_", ""))
                    # print(channels)
                    basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
                    try: 
                        information = pd.read_csv('%s/%sChannel_info_%s.csv' %(path, day, basename_without_ext))
                        print(information)
                    except:
                        print('Channel Information not available')
                        # Create empty dataframe with information for all channels
                        information = pd.DataFrame() #columns=['ch_string', 'intan_ch', 'Z_magnitude', 'Z_phase'])
                else:
                    print('ERROR: You have selected a wrong file, try again')
                    sys.exit()
    
    else:
        if load_multiple_files:
            print('Load multiple files')
            # glob allows to load all files in the folder
            files = glob.glob(path+'/*.%s'%fileType, recursive=True)
            if verbose:
                print(fileType)
                print(files)
            # Create arrays to store all the data
            amp_data = [] 
            time = []            
            # Run over all files
            for count,file in enumerate(files):
                print(count)
                if fileType == 'rhs':
                    print('Loading rhs files from %s' %file)
                    data = read_rhs_data(file, verbose=verbose)
                # else: 
                #     file = file.replace("\\", "/" )
                #     from load_intan_rhd_format.load_intan_rhd_format import read_data
                #     data = read_data(file) #'C:/Users/ampar/OneDrive - University of Cambridge/4. pyNeural/datasets/pisa/pigs/2023_02_07_pig/record_230207_142309/record_230207_142309.rhd')
                if verbose:
                    print(data)
                # Sampling frequency 
                fs = data['frequency_parameters']['amplifier_sample_rate']
                # Create dataframe
                # As channels not in columns format then transpose
                if (np.shape(data['amplifier_data'])[1]>np.shape(data['amplifier_data'])[0]):
                    # Downsample 
                    print('Downsampling with factor: %s' %downsample)
                    new_amp_data = data['amplifier_data'].transpose()
                    new_amp_data = new_amp_data[::downsample]
                    # Transpose data to have channels as columns
                    amp_data.append(new_amp_data)
                    new_time = data[time_key].transpose() 
                    new_time = new_time[::downsample]       #start:stop:step
                    time.append(new_time)
                else:
                    new_amp_data = data['amplifier_data']
                    new_amp_data = new_amp_data[::downsample]
                    amp_data.append(new_amp_data)
                    new_time = data[time_key].transpose()
                    new_time = new_time[::downsample]       #start:stop:step
                    time.append(new_time)
            amp_data = np.concatenate(amp_data, axis=0)
            time = np.concatenate(time, axis=0)
            filepath_init = files[0]
            basename_init = os.path.splitext(os.path.basename(filepath_init))[0]
            filepath_end = files[-1]
            basename_end = os.path.splitext(os.path.basename(filepath_end))[0]
            basename_without_ext = basename_init+'_'+basename_end[-6:]
        else:  
            # Open GUI for selecting file
            filepath_init = askopenfile(initialdir=path, title="Select data file (.mat or .rhs)",
                                    filetypes=[("rhs", ".rhs"), ("matlab", ".mat")])
            filepath_init = filepath_init.name
            # Load data
            if filepath_init.endswith('.mat'):
                #print('Loading mat files from %s' %filepath_init)
                data = scipy.io.loadmat(filepath_init)
                # Sampling frequency 
                fs = data['fs']
                # Extract only value from nested array
                fs = fs[0][0]
            elif filepath_init.endswith('.rhs'):
                #print('Loading rhs files from %s' %filepath_init)
                data = read_rhs_data(filepath_init,verbose=verbose)
                # Sampling frequency 
                print(data['frequency_parameters'])
                fs = data['frequency_parameters']['amplifier_sample_rate']
            elif filepath_init.endswith('.csv'):
                print('ERROR: csv file selected. Please choose a .mat or .rhs file')
            
            # If channels not in columns then transpose
            if (np.shape(data['amplifier_data'])[1]>np.shape(data['amplifier_data'])[0]):
                # Transpose data to have channels as columns
                amp_data = data['amplifier_data'].transpose()
                amp_data = amp_data[::downsample]
                time = data[time_key].transpose()
                time = time[::downsample]       #start:stop:step
            else:
                amp_data = data['amplifier_data']
                amp_data = amp_data[::downsample]
                time = data[time_key]
                time = time[::downsample]       #start:stop:step
            basename_without_ext = os.path.splitext(os.path.basename(filepath_init))[0]

        # Downsample frequency
        #print('Downsampling with factor: %s' %downsample)
        print('time: %s' %len(time))
        print('data: %s' %len(amp_data))
        
        #amp_data = amp_data[::downsample]
        fs = fs/downsample

        # General to all raw files loading
        # Set time interval
        if dur is None:
            stop = len(amp_data)
        else:
            stop = int(start + dur)

        start = int(start)

        amp_data = amp_data[start:stop]
        time = time[start:stop]

        fs = fs / downsample
        # Create dataframe to store voltage data
        neural = pl.DataFrame() 

        # Create dataframe with information for all channels
        information = pl.DataFrame() #columns=['ch_string', 'intan_ch', 'Z_magnitude', 'Z_phase'])

        channels_info = data.get('amplifier_channels', [])

        selected_indices = []
        column_names = []
        intan_ch = []
        Z_magnitude = []
        Z_phase = []

        for i, el in enumerate(channels_info):
            if el['port_name'] == port:
                selected_indices.append(i)
                ch_name = f"ch_{el['native_order']}"
                column_names.append(ch_name)

                intan_ch.append(el['native_order'])
                Z_magnitude.append(el['electrode_impedance_magnitude'] / 1000)
                Z_phase.append(el['electrode_impedance_phase'])

        # Extract only selected channels
        amp_selected = amp_data[:, selected_indices]

        # Build dict for Polars
        data_dict = {
            column_names[i]: amp_selected[:, i].astype("float32")
            for i in range(len(column_names))
        }

        # Add time column (convert seconds → datetime ns)
        data_dict["time"] = (time * 1e9).astype("int64")

        neural = pl.DataFrame(data_dict)

        # Convert to proper Datetime type
        neural = neural.with_columns(
            pl.col("time").cast(pl.Datetime("ns"))
        )

        # -------------------------------------------------
        # Save Parquet
        # -------------------------------------------------
        p = Path(filepath_init)

        basename_without_ext = p.stem          # "recording"
        parent_dir = p.parent                  # directory path
        subfolder = parent_dir / basename_without_ext
        subfolder.mkdir(parents=True, exist_ok=True)

        output_path = subfolder/ f"{basename_without_ext}_{port}.parquet"

        print(f"Saving data into: {output_path}")
        neural.write_parquet(output_path)

        # -------------------------------------------------
        # Channel Information (Polars)
        # -------------------------------------------------
        information = pl.DataFrame({
            "ch_string": column_names,
            "intan_ch": intan_ch,
            "Z_magnitude_KOhms": Z_magnitude,
            "Z_phase": Z_phase,
        })

        info_path = subfolder / f"{day}Channel_info_{basename_without_ext}_{port}.csv"
        

        information.write_csv(info_path)

        # -------------------------------------------------
        # Optional impedance saving
        # -------------------------------------------------
        if day != "":
            path_rat = os.path.dirname(os.path.dirname(path))
            print(f"Saving impedance per day into: {path_rat}")
            save_impedance_data(intan_ch, Z_magnitude, day, path_rat)

    return neural, fs, basename_without_ext, information

def load_all_ports_rhs(path,fileType='rhs', downsample=1,verbose=1):
    """
    Load all .rhs files in a folder and return a dict with port-separated DataFrames and metadata.
    
    Returns:
        df_by_port (dict): Port-wise DataFrames with channels as columns.
        fs (float): Sampling frequency.
        channel_info (list): List of dicts with channel metadata.
        full_df (DataFrame): Complete DataFrame with all channels.
    """
    
    # Gather all files
    files = sorted(glob.glob(os.path.join(path, f'*.{fileType}'), recursive=True))
    if verbose:
        print(f"Found {len(files)} {fileType} files in: {path}")
        print(files)
    
    amp_data = []
    time_data = []
    fs = None
    all_channel_info = None

    # Loop through each file
    for count, file in enumerate(files):
        print(f"Loading file {count+1}/{len(files)}: {file}")
        data = read_rhs_data(file, verbose=verbose)
        
        # Sampling frequency (assume constant)
        if fs is None:
            fs = data['frequency_parameters']['amplifier_sample_rate']
        
        # Channel metadata (assume constant)
        if all_channel_info is None:
            all_channel_info = data['amplifier_channels']
        
        # Amplifier data
        amp = data['amplifier_data'].T  # shape: [samples, channels]
        amp = amp[::downsample]
        amp_data.append(amp)

        # Time vector
        t = data['t'] if 't' in data else data['t_amplifier']
        t = t[::downsample]
        time_data.append(t)

    # Concatenate across files
    amp_data = np.vstack(amp_data)  # shape: [total_samples, channels]
    time_data = np.concatenate(time_data)

    # Create column labels from amplifier_channels
    channel_names = [f"{ch['port_name']}_{ch['native_channel_name']}" for ch in all_channel_info]
    port_names = [ch['port_name'] for ch in all_channel_info]

    df_all = pd.DataFrame(amp_data, columns=channel_names)
    df_all['time'] = time_data

    # Split into port-wise DataFrames
    df_by_port = {}
    for port in set(port_names):
        cols = [f"{ch['port_name']}_{ch['native_channel_name']}" for ch in all_channel_info if ch['port_name'] == port]
        df_port = df_all[cols + ['time']].copy()
        df_by_port[port] = df_port

    return df_by_port, fs, all_channel_info, df_all


def save_impedance_data(intan_ch, Z_magnitude, day, path_rat):
    """
    Saves impedance data to a CSV file, ensuring proper handling of different channel lengths.

    Args:
        intan_ch (np.ndarray): Array of channel indices.
        Z_magnitude (np.ndarray): Array of corresponding impedance magnitudes.
        day (str): Day string to use as a column name.
        csv_filename (str): Path to the CSV file.
    """
    # Path and filename (replace with your actual values)
    filename = "Z_days.csv"

    # Check if the CSV file exists
    exists = os.path.exists(os.path.join(path_rat, filename))

    ## Create empty DataFrame with day column if file doesn't exist
    if not exists:
        df = pd.DataFrame()
        df['intan_ch'] = np.arange(0,32,1)
    else:
        # Load existing DataFrame
        df = pd.read_csv(os.path.join(path_rat, filename))

    # Handle different channel lengths between days
    max_channels = 32
    padding_value = 'nan'

    # Create the new column with appropriate name (replace "ChX" with your desired name)
    if day not in df.columns:
        df['%s'%day] = pd.Series([] * max_channels)
    else:
        print('Day already saved in dataframe. Re-run or change day')
    # Populate the new column with Z_magnitude values at specified positions (intan_ch)
    for i, intan_ch in enumerate(intan_ch):
        df.loc[intan_ch, '%s'%day] = Z_magnitude[i]

    # Store DataFrame as CSV
    df.to_csv(os.path.join(path_rat, filename), index=False)

def load_data_dask(path, start=0, dur=None, port='Port B', load_from_file=False, load_multiple_files=False, downsample=1, verbose=1):
    """This method loads...

    .. note: This probably should go into a library that could be 
             inside the the datasets folder. Then you could import
             it with something like:

               from datasets import load_setup

    Parameters
    ----------
    path :      [string-like] The path for the corresponding setup
    start:      [int] sample onset
    dur  :      [int] duration in samples
    channels:   [array] channels to be selected. If a single port is used then it's always (0, 32), if two ports then port A (0,32) and port B(32, 64)
    load_from_file: [boolean] By default False loads from mat o rhs files, otherwise loads 
                    a previously stored csv or pkl with the dataframe
    downsample: [int] (default: 1) downsampling factor.
    verbose:    [int] signal to display text information (default 1 - show text)
    
    Returns
    -------
    neural:     [dataframe] Index is the time in DateTime, one column for each channel
    fs:         [float] Sampling frequency
    basename_without_ext: [string] name of the file without the extension. For storing purposes
    channels:   [array of ints] Available channels from amplifier data in rhs file

    """      

    # Extract directory of current path
    # dir_name = os.path.dirname(path)

    # Open GUI for selecting file
    Tk().withdraw()  # keep the root window from appearing
    print(path)
    # If data has been previously stored
    if load_from_file:
        filepath = askopenfile(initialdir=path, title="Select previously stored data file", 
                                filetypes=[("recording", ".csv .pkl")])
        filepath = filepath.name
        print('Loading from file %s' %filepath)
        channels = []
        # Load from csv: computationally expensive
        if filepath.endswith('.csv'):
            neural_pd = pd.read_csv(filepath)
            neural = dd.from_pandas(neural_pd, npartitions=10)
            # Check the file is a data file
            if 'time' in neural.columns:
                # Remove 'time' column and remake it as index (otherwise it's imported as String and not Dataframe)
                neural = neural.drop(columns=['time'])
                neural.index = pd.DatetimeIndex(neural.seconds * 1e9)
                neural.index.name = 'time'

                # Set time interval
                print(start)
                if dur is None:
                    stop = len(neural)
                else:
                    stop = int(start + dur)
                print('stop: %s' %stop)
                start = int(start)

                neural = neural[start:stop]

                # Get Sampling frequency
                fs = 1/(neural['seconds'][2]-neural['seconds'][1])
                print(fs)

                # Downcast it type float64 
                for col in neural.columns:
                    if col.startswith('ch_'):
                        neural[col] = neural[col].astype('float32')

                basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
            else:
                print('ERROR: You have selected a wrong file, try again')
                sys.exit()
        #Load from pickle: much faster
        elif filepath.endswith('.pkl'):
            neural_pd = pd.read_pickle(filepath)
            neural = dd.from_pandas(neural_pd, npartitions=10)

            # Check the file is a data file
            if neural.index.name == 'time':
                # Set time interval
                print(start)
                if dur is None:
                    stop = len(neural)
                else:
                    stop = int(start + dur)
                print('stop: %s' %stop)
                start = int(start)
                print(neural['seconds'].compute().iloc[2])
                print(neural.compute())
                neural = neural.compute().iloc[start:stop]

                # Get Sampling frequency
                fs = 1/(neural['seconds'][2]-neural['seconds'][1])
                print(fs)

                # Downcast it type float64 
                for col in neural.columns:
                    if col.startswith('ch_'):
                        neural[col] = neural[col].astype('float32')
                        channels.append(col.replace('ch_', ''))
                print(channels)
                basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
            else:
                print('ERROR: You have selected a wrong file, try again')
                sys.exit()
    
    else:
        if load_multiple_files:
            """ Previous code"
            # Load first and last files
            filepath_init = askopenfile(initialdir=path, title="Select initial recording", filetypes=[("rhs", ".rhs")])
            filepath_end = askopenfile(initialdir=path, title="Select final recording", filetypes=[("rhs", ".rhs")])
            # Get time for each file
            filepath_init = filepath_init.name
            basename_init = os.path.splitext(os.path.basename(filepath_init))[0]
            init_time = int(basename_init[-6:])
            filepath_end = filepath_end.name
            basename_end = os.path.splitext(os.path.basename(filepath_end))[0]
            end_time = int(basename_end[-6:])
            # Get time difference (Work with datetime)
            FMT = '%H%M%S'
            tdelta = datetime.datetime.strptime(basename_end[-6:], FMT) - datetime.datetime.strptime(basename_init[-6:], FMT)
            dur_min = int(tdelta.seconds/60)
            # Create arrays to store all the data
            amp_data = [] 
            time = []            
            # Run over all files
            for m in np.arange(dur_min+1):
                new_time = datetime.datetime.strptime(basename_init[-6:], FMT) + datetime.timedelta(0,int(m)*60)
                file = path+'/'+basename_init[:-6]+new_time.strftime(FMT)+'.rhs'
                data = read_data(file, verbose=verbose)
                # Sampling frequency 
                fs = data['frequency_parameters']['amplifier_sample_rate']
                
                # Create dataframe
                # If channels not in columns then transpose
                if (np.shape(data['amplifier_data'])[1]>np.shape(data['amplifier_data'])[0]):
                    # Transpose data to have channels as columns
                    amp_data.append(data['amplifier_data'].transpose())
                    time.append(data['t'].transpose())
                else:
                    amp_data.append(data['amplifier_data'])
                    time.append(data['t'])
            """
            # glob allows to load all files in the folder
            files = glob.glob(path+'/*.rhs', recursive=True)
            # Create arrays to store all the data
            amp_data_np = []
            amp_data = da.empty(1) 
            time_np = []            
            time = da.empty(1)
            # Run over all files
            for file in files:
                data = read_data(file, verbose=verbose)
                # Sampling frequency 
                fs = data['frequency_parameters']['amplifier_sample_rate']
                # Create dataframe
                # As channels not in columns format then transpose
                if (np.shape(data['amplifier_data'])[1]>np.shape(data['amplifier_data'])[0]):
                    # Downsample 
                    print('Downsampling with factor: %s' %downsample)
                    new_amp_data = data['amplifier_data'].transpose()
                    new_amp_data = new_amp_data[::downsample]
                    # Transpose data to have channels as columns
                    #amp_data.append(new_amp_data)
                    amp_data = np.append(amp_data,new_amp_data)
                    new_time = data['t'].transpose()
                    new_time = new_time[::downsample]       #start:stop:step
                    #time.append(new_time)
                    time = np.append(time,new_time)
                else:
                    new_amp_data = data['amplifier_data']
                    new_amp_data = new_amp_data[::downsample]
                    amp_data.append(new_amp_data)
                    new_time = data['t'].transpose()
                    new_time = new_time[::downsample]       #start:stop:step
                    time.append(new_time)
            amp_data = np.concatenate(amp_data, axis=0)
            time = np.concatenate(time, axis=0)
            filepath_init = files[0]
            basename_init = os.path.splitext(os.path.basename(filepath_init))[0]
            filepath_end = files[-1]
            basename_end = os.path.splitext(os.path.basename(filepath_end))[0]
            basename_without_ext = basename_init+'_'+basename_end[-6:]
        else:  
            # Open GUI for selecting file
            filepath_init = askopenfile(initialdir=path, title="Select data file (.mat or .rhs)",
                                    filetypes=[("rhs", ".rhs"), ("matlab", ".mat")])
            filepath_init = filepath_init.name
            # Load data
            if filepath_init.endswith('.mat'):
                #print('Loading mat files from %s' %filepath_init)
                data = scipy.io.loadmat(filepath_init)
                # Sampling frequency 
                fs = data['fs']
                # Extract only value from nested array
                fs = fs[0][0]
            elif filepath_init.endswith('.rhs'):
                #print('Loading rhs files from %s' %filepath_init)
                data = read_data(filepath_init,verbose=verbose)
                # Sampling frequency 
                print(data['frequency_parameters'])
                fs = data['frequency_parameters']['amplifier_sample_rate']
            elif filepath_init.endswith('.csv'):
                print('ERROR: csv file selected. Please choose a .mat or .rhs file')
            
            # If channels not in columns then transpose
            if (np.shape(data['amplifier_data'])[1]>np.shape(data['amplifier_data'])[0]):
                # Transpose data to have channels as columns
                amp_data = data['amplifier_data'].transpose()
                amp_data = amp_data[::downsample]
                time = data['t'].transpose()
                time = time[::downsample]       #start:stop:step
            else:
                amp_data = data['amplifier_data']
                amp_data = amp_data[::downsample]
                time = data['t']
                time = time[::downsample]       #start:stop:step
            basename_without_ext = os.path.splitext(os.path.basename(filepath_init))[0]

        # Downsample frequency
        #print('Downsampling with factor: %s' %downsample)
        print('time: %s' %len(time))
        print('data: %s' %len(amp_data))
        #amp_data = amp_data[::downsample]
        fs = fs/downsample

        # General to all raw files loading
        # Set time interval
        if dur is None:
            stop = len(amp_data)
        else:
            stop = int(start + dur)
        start = int(start)

        # Create dataframe
        neural_df = pd.DataFrame() 

        # Add column names
        columns = []
        channels = []
        for i, el in enumerate(data['amplifier_channels']):
            # data['amplifier_channels'] contains info as in this example:
            # {'port_name': 'Port A', 'port_prefix': 'A', 'port_number': 1, 'native_channel_name': 'A-030', 'custom_channel_name': 'A-030', 
            # 'native_order': 30, 'custom_order': 30, 'chip_channel': 14, 'board_stream': 1, 'electrode_impedance_magnitude': 6839.2158203125, 
            # 'electrode_impedance_phase': -59.66900634765625}   
            if el['port_name'] == port:
                columns.append('ch_%s' %(el['custom_order']))   # Array of strings with available channels in the format 'ch_x'
                channels.append(el['custom_order'])             # Array of ints with the number of the available channels
                neural_df['ch_%s' %(el['custom_order'])] = amp_data[:,i]
        print(columns)
        print(channels)

        # Create dataframe
        #neural = pd.DataFrame(data=amp_data[, columns=columns) #[:,channels]
        neural_df['seconds'] = time

        # Set datetime index
        neural_df.index = pd.DatetimeIndex(neural.seconds * 1e9)
        neural_df.index.name = 'time'
        
        neural = dd.from_pandas(neural_df, npartitions=10)


        neural = neural.iloc[start:stop]

        # Downcast
        for col in neural.columns:
            if col.startswith('ch_'):
                neural[col] = neural[col].astype('float32')

        
        # To avoid loading the data and creating the dataframe every time save it as csv 
        #print('Saving data into: %s/%s.csv' %(path, basename_without_ext))
        #neural.to_csv(r'%s/%s.csv' %(path, basename_without_ext))

        print('Saving data into: %s/%s_%s.pkl' %(path, basename_without_ext,port))
        neural.compute().to_pickle(r'%s/%s_%s.pkl' %(path, basename_without_ext, port))

    # Return
    return neural, fs, basename_without_ext, channels


if __name__ == '__main__':
    # GUI folder
    from tkinter import Tk
    from tkinter.filedialog import askdirectory, asksaveasfilename
    # Type of experiment
    type_exp = 'stimulation'

    # Date of experiment
    rat = 'rat_008'

    # ----------------
    # Load data
    # ----------------
    # Select directory
    Tk().withdraw()  # keep the root window from appearing
    path = askdirectory(initialdir="../datasets/%s/%s"
                        % (type_exp, rat), title="Select folder")

    path_bsamples = '%s/_bsamples.csv' % path
    data = load_sampletime(path_bsamples, fs=300, nsamples=2250000, verbose=1)
    print(data.tails)
    print(data.artery[681299])
    plt.plot(data.seconds, data.clot)
    plt.show()

    """
    # Load dataframes
    ecg, glucose, stimuli, fs = load_setup(path)

    # Concatenate them
    concat = pd.concat((ecg, glucose, stimuli), axis=1)

    # Show
    print(concat.head(10))
    """
