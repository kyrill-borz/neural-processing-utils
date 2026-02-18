# Libraries
import os
import sys
import json
import time
import datetime
import pycwt
import numpy as np
from scipy.stats import ks_2samp, expon
import numpy as np
import scipy as sp
import pandas as pd
import polars as pl
import seaborn as sns
import sklearn as sk
import imageio
import itertools
import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
import pylab
#import umap
import umap.umap_ as umap
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
# from tkinter import simpledialog
from sklearn import metrics
import statistics
import scipy.stats
from sklearn import preprocessing
# Import listed colormap
from matplotlib.colors import ListedColormap
import matplotlib.dates as md
from pathlib import Path

import dask.dataframe as dd


#import plotly.io as plt_io
#import plotly.graph_objects as go

# Scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy import ndimage

# TKinter for selecting files
from tkinter import Tk	 # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfile

# Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from math import pi
from bokeh.layouts import gridplot

# Superparamagnetic clustering
from spclustering import SPC #, plot_temperature_plot  https://github.com/ferchaure/SPC

# Add my module to python path
sys.path.append("../")

# Own libraries
from utils.load_data import load_setup, load_matfiles, load_data_multich, load_bsamples,load_data_dask, load_bsamples_start_end
from utils import *
# from visualization.graphics.plots import *
# from processing.filter import FIR_smooth
# from visualization.SC_topo import *
from matplotlib.ticker import FuncFormatter
from utils.autofilter import adaptive_filter

def hide_tick_labels(value, pos):
		return ""

class SpikeResult:
    channel: str
    indices: np.ndarray
    times: np.ndarray
    waveforms: np.ndarray | None = None

class Recording:
	"""Class for pre-processing and extracting spikes and HR from electroneurograms (ENG)"""
	def __init__(self, neural, fs, length, map_array, filename, information): # intan_ch, Z_magnitude, Z_phase):
		"""Constructor of neurogram object. 
		Takes parameters from classmethod open_record, and also initialises other parameters
		
		Parameters
		------------
		neural: dataframe 
		fs: scalar
		length: lenth of neural dataframe 
		map_array:
		filename:
		intan_ch: [list of ints] Available channels from amplifier data in rhs file

		Return
		--------
		self: object that contains all the parameters

		"""
		#self.recording = neural.neural_b
		self.recording = neural	# Initialise with dataframe
		self.recording.name = 'original'
		self.original = neural   # keep a copy of the original 
		self.original.name = 'original'   # keep a copy of the original 
		self.information = information # Contains ch_string, intan_ch, Z_magnitude, Z_phase
		#self.intan_ch = intan_ch   # All available columns in the intan files (previously column_ch)
		self.fs = fs
		self.filename = filename
		self.threshold = []
		self.wavelet_results = {}
		self.ch_loc = []	# Device electrode number corresponding to intan channels (to save location of electrode as integer): 1 to 32 to differenciate form INTAN ch
		self.filter_ch = [] # Selected intan channels to analyse (0-31)
		self.Z_magnitude = [] # Impedane magnitude of 
		self.Z_phase = [] # Impedane phase of selected channels
		self.map_array=map_array  # Intan channels corresponding to electrodes numbered 1 to 32: intan ch corresponding to electrode 1 = map_array[0]
		self.length = length
		self.signal_coded = []
		self.apply_filter = []
		self.detect_method = []											
		self.thresh_type = []	
		self.channels = []
		self.results=[]

	@classmethod
	def open_record(cls, path,start, dur=None, load_multiple_files=False, downsample=1, port='Port B', map_path=None, pig=False, day='', verbose=1):
		"""
		Called by constructor method to load neural recordings and electrode map 
		
		Parameters
		------------
		path : 		[string-like] The path for the corresponding setup
		start: 		[int] sample onset
		dur  : 		[int] duration in samples
		load_from_file: [boolean] By default False loads from mat o rhs files, otherwise loads a previously stored csv or pkl with the dataframe
		load_multiple_files: [boolean] load from multiple rhs files or single file (default: False - load single file)
		downsample: [int] (default: 1) downsampling factor.
		channels: 	[array] channels to be selected. If a single port is used then it's always (0, 32), if two ports then port A (0,32) and port B(32, 64)
		map_path: 	[string] path to csv where electrode map is stored
		verbose:	[int - 0,1] signal to display text information (default 1 - show text, 0 - don't show) 

		Return
		--------
		neural: 	[dataframe] Index is the time in DateTime, one column for each channel 
		fs:  		[float] Sampling frequency
		length: 	[int] number of samples of neural dataframe (number of rows). 
		map_array: 	[numpy array] 1D array with the corresponding intan channels for linear electrode device (1-31 electrodes): map_array[0] is intan channel corresponding to electrode 1
		basename_without_ext: [string] name of the file without the extension. For storing purposes
		intan_ch:   [list of ints] Available channels from amplifier data in rhs file. Outdated, now contained in 'information['intan_ch']'

		"""

		map_array = {}
		# Load dataframes
		if pig:
			# Read data
			neural = pd.read_csv(path+port+'.csv')
			fs = 1/(neural.seconds[2]-neural.seconds[1])
			neural.index = pd.DatetimeIndex(neural.seconds * 1e9)
			neural.index.name = 'time'
			neural = neural.iloc[start:start+dur]
			basename_without_ext = port
			 
		else:
			fileType = Path(path).suffix.lower().replace('.', '')
			if fileType in ['rhs', 'mat']:
				load_from_file = False
			else:
				load_from_file = True
			print("open record")
			neural,fs, basename_without_ext, information = load_data_multich(path, start=start, dur=dur, port=port,  #intan_ch, Z_magnitude, Z_phase 
												load_from_file=load_from_file,
												load_multiple_files=load_multiple_files,
												fileType=fileType, 
												downsample=downsample,
												day=day, # Added Feb 2024 to account for chonic data
												verbose=verbose)
		'''
		neural,fs, basename_without_ext, intan_ch = load_data_dask(path, start=start, dur=dur, port=port, 
											load_from_file=load_from_file,
											load_multiple_files=load_multiple_files,
											downsample=downsample,
											verbose=verbose)
		'''
		print(neural)
		length = len(neural)
		print(length)
		#print(information)

		# Load electrode map
		if map_path is None:
			Tk().withdraw()  # keep the root window from appearing
			map_filepath = askopenfile(initialdir=path, title="Select electrode map .csv",
										filetypes=[("map", ".csv")])
		else: 
			map_filepath=map_path
		map_array = pd.read_csv(map_filepath, header=None)
		map_array = map_array.to_numpy()
		map_array = map_array.flatten()
		
		try:	
			np.shape(map_array)[0]
		except:
			print('If map_array is 1D, it needs to be a row. Transposing...')
			map_array=map_array.transpose()
		else: 
			map_array=map_array
		
		print("Data loaded succesfully.")
		print('Sampling frequency: %s' %fs)
		print('Recording length: %s(samples), %s(s): ' %(length, length/fs))
		return cls(neural, fs, length, map_array, basename_without_ext, information) #intan_ch, Z_magnitude, Z_phase)
	
	def startAnalysisGui(self, options_filter, options_detection, options_threshold):
		# -----------------
		# GUI for inputs
		# -----------------
		root = Tk()
		root.title('Select your analysis preferences')
		root.geometry("400x400")
		root.eval('tk::PlaceWindow . center')
		# e = Entry(root, width=400, borderwidth=5) # bg="blue", fg="white"

		def assign():
			self.apply_filter = filterCombo.get() 	
			self.detect_method = detectionCombo.get()											
			self.thresh_type = thresholdCombo.get()
			if	entry.get():
				self.channels = entry.get().split(",") 
			else:
				print('Please select channel(s) to analyse')
				sys.exit()
			
			print('SELECTED CONFIGURATION:')
			print('Filter: %s'%filterCombo.get())
			print('Detection: %s'%detectionCombo.get())
			print('Threhold type: %s'%thresholdCombo.get())
			print('Channels: %s' %self.channels)
		
			root.quit()
			root.destroy()

		# Combo Boxes
		filterLabel = Label (root, text="Select filter: ")
		filterLabel.pack()
		filterCombo = ttk.Combobox(root, value=options_filter)
		filterCombo.current(0)
		filterCombo.pack(pady=10)

		detectionLabel = Label (root, text="Select detection method: ")
		detectionLabel.pack()
		detectionCombo = ttk.Combobox(root, value=options_detection)
		detectionCombo.current(0)
		detectionCombo.pack(pady=10)

		thresholdLabel = Label (root, text="Select threshold type: ")
		thresholdLabel.pack()
		thresholdCombo = ttk.Combobox(root, value=options_threshold)
		thresholdCombo.current(0)
		thresholdCombo.pack(pady=10)
		
		# Enter channels
		channelLabel = Label (root, text="Enter channels to analyse separated by ',' or \"all\": ")
		channelLabel.pack()
		entry = Entry(root, width=50,) # bg="blue", fg="white"
		entry.pack()
		
		# Create Button
		myButton = Button(root, text="Start analysis", command=assign)
		myButton.pack(pady=25) 

		root.mainloop()

	def select_channels(self, channels):
		"""
		Method to select which channels to analyse 
		
		Parameters
		------------
		channels: 		['all' or list of numbers] list of selected intan channels to be analysed
		self.information['intan_ch']:	[list] list of native intan channels from amplifier stored in rhs file (same as channels in load)
		self.map_array:	[numpy array] 1D array with the corresponding intan channels for linear electrode device (1-32 electrodes): map_array[0] is intan channel corresponding to electrode 1

		Return
		--------
		self: object updated with channels information
			ch_loc: 	[list of int] list with electrodes locations corresponding to the selected intan channels (inverse of map_array: ch_loc[0] is electrode corresponding to intan ch0 )
			filter_ch:	[list of string] list with the selected intan channels in string mode (starting in 'ch_')

		"""
		print(channels)
		if 'all' in channels:
			nchannels = self.information['intan_ch'] #self.intan_ch
			# Commented below 29/03/22 because it will only work if all intan channels are available, otherwise I'll need to change the map
			#self.ch_loc = self.intan_ch # np.arange(len(self.recording.columns[:-1]))
			#self.filter_ch = np.asarray(self.recording.columns[:-1]) # ['ch_%s'%int(c) for c in self.map_array if ~np.isnan(c)]
		else:
			nchannels = channels
		print(nchannels)


		# Need to initialise to remake channels whenever the function is called with new selected channels 
		self.ch_loc = []
		self.filter_ch = []
		self.Z_magnitude = []

		for i, ch in enumerate(nchannels):
			ch = int(ch)
			if ch < 0 or (ch not in self.map_array):
				print("Channel not found")
				sys.exit() 
			else:
				self.ch_loc.append(np.where(self.map_array == ch)[0][0]+1) # +1 to differeciate electrodes (1-32) from intan (0-31)
				self.filter_ch.append('ch_%s'%ch)
				try:
					self.Z_magnitude.append(self.information['Z_magnitude Kohms'].iloc[i])
				except:
					print('no Z_magnitude')
		

	def set_gain(self, gain=1):
		"""
		Apply amplitude gain to recording

		Parameters
		------------
		gain: [float] Ratio for gain

		Return
		------------
		Recording dataframe updated with gain applied to each of the channels selected 
		"""
		self.recording = self.recording.apply(lambda x: (x * gain) if x.name in self.filter_ch else x)
	
	def set_bpm(self, bpm):
		"""
		Method to set heart rate (beats per minute)

		Parameters
		------------
		bpm: [float] beats for minute

		Return
		------------
		Recording object updated with the bpm parameter 
		"""
		self.bpm=bpm

	def define_stimulation(self, time2onset, interval, code_template=[0]):
		"""
		TO DO: write description
		"""
		samples_interval = int(interval * self.fs)
		samples2onset = int(time2onset * self.fs - samples_interval/2)
		num_groups = len(code_template)
		num_intervals = int((self.length - samples2onset)/samples_interval)
		signal_coded = []
		signal_coded.extend(np.ones(samples2onset)*np.nan)

		g = 0
		for i in np.arange(num_intervals+2):
			signal_coded.extend(np.ones(samples_interval)*code_template[g])
			if g == (num_groups-1):
				g = 0
			else: 
				g = g+1
		self.signal_coded = np.array(signal_coded)

	# ---------------------------------------------------------------------------
	# Filtering methods 
	# ---------------------------------------------------------------------------
	def filter(self, signal2filt,filtername, channels=['ch_27'], **kargs, ):
		"""
		Method to apply filtering to recordings (ENG)
		Note that despite the whole dataframe is passed, the algorithm only applies to the selected channels (filter_ch)

		Parameters
		------------
		signal2filt: [dataframe] signals to filter (columns in dataframe structure)
		filtername:	 [string] name of the filter to apply {'None', 'butter', 'fir', 'notch'}
		kargs:		 [dict] specific parameters for for the filters

		Returns
		------------
		self.filtered: [dataframe] updare the recording object with a parameter that is a dataframe with the results of the filtering

		"""
		if filtername=='No Filter':
			self.filtered = self.recording
			print('No filter applied!')
			pass
		elif filtername=="Automatic":
			
			print('Applying automatic filter selection')
			self.filtered, metadata = adaptive_filter(
				self.original,
				fs=self.fs,
				channels=self.filter_ch
			)
			

		elif filtername=='Butterworth':
			print('Applying Butterworth bandpass')
			# SOS filter used in gut, VN acute, and all analysis until Feb 2024. It's causal and therefore introduces a delay
			#-----------------------------
			# Configure butterworth filter
			# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html#scipy.signal.sosfilt
			#b, a = signal.butter(**kargs)
			#w, h = signal.freqs(b, a)
			#filt_config['butter']['Wn'] = fnorm(filt_config['W'], fs=record.fs).tolist() # The critical frequency or frequencies.
								# Same as doing (filt_config['W']/(record.fs/2)).tolist()
			#kargs['butter']['Wn'] = kargs['W']
			print(kargs)
			kargs['fs'] = self.fs
			sos = signal.butter(**kargs, output='sos')  # Coefficients for SOS filter

			# Filter signal: high pass (cutoff at 100Hz)
			#----------------------------------------------
			# Old filter
			#self.recording[self.filter_ch] = signal.lfilter(b, a, self.recording[self.filter_ch])
			#self.recording[self.filter_ch] = signal.sosfilt(sos, self.recording[self.filter_ch])
			#----------------------------------
			
			self.filtered = signal2filt.with_columns(
				[
					pl.col(col)
					.map_batches(lambda s: signal.sosfilt(sos, s.to_numpy()))
					.alias(col)
					for col in self.filter_ch
				]
			)
		elif filtername=='Lowpass':
			print('Applying low pass butter')
			# SOS filter used in gut, VN acute, and all analysis until Feb 2024. It's causal and therefore introduces a delay
			#-----------------------------
			# Configure butterworth filter
			# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html#scipy.signal.sosfilt
			#b, a = signal.butter(**kargs)
			#w, h = signal.freqs(b, a)
			#filt_config['butter']['Wn'] = fnorm(filt_config['W'], fs=record.fs).tolist() # The critical frequency or frequencies.
								# Same as doing (filt_config['W']/(record.fs/2)).tolist()
			#kargs['butter']['Wn'] = kargs['W']
			print(kargs)
			kargs['fs'] = self.fs
			sos = signal.butter(**kargs, output='sos')  # Coefficients for SOS filter

			# Filter signal: high pass (cutoff at 100Hz)
			#----------------------------------------------
			# Old filter
			#self.recording[self.filter_ch] = signal.lfilter(b, a, self.recording[self.filter_ch])
			#self.recording[self.filter_ch] = signal.sosfilt(sos, self.recording[self.filter_ch])
			#----------------------------------
			# self.filtered = signal2filt.with_columns(
			# 	[
			# 		pl.col(col)
			# 		.map_batches(lambda s: signal.sosfilt(sos, s.to_numpy()))
			# 		.alias(col)
			# 		for col in self.filter_ch
			# 	]
			# )
			self.filter_ch = channels
			print(self.filter_ch)
			df = signal2filt.select(self.filter_ch)

			self.filtered = signal2filt.with_columns(
				[
					pl.Series(
						name=col,
						values=signal.sosfilt(sos, df[col].to_numpy())
					)
					for col in self.filter_ch
				]
			)
			print(self.filtered)

		elif filtername=='butter_non_causal':
			#----------------------------------
			# New non-causal SOS filter
			# Replaced signal.sosfilt with signal.filtfilt: This function filters the signal in both the forward and reverse directions, effectively creating a zero-phase, non-causal filter.
			# Passed sos twice as arguments: filtfilt requires both the forward and reverse filter coefficients, which are identical for SOS filters.
			# Filter signal: high pass (cutoff at 100Hz)

			print(kargs)
			kargs['fs'] = self.fs
			sos = signal.butter(**kargs, output='sos')  # Coefficients for SOS filter
			self.filtered = signal2filt.apply(lambda x: signal.filtfilt(sos, sos, x)  # Apply filtfilt
												if x.name in self.filter_ch else x)

		elif filtername=='fir':
			print(self.filter_ch)
			self.filtered = signal2filt.apply(lambda x: FIR_smooth(x, **kargs) 
													if x.name in self.filter_ch else x)
		elif filtername=='notch':
			self.filtered = signal2filt.apply(lambda x: self.iir_notch(x, **kargs)
														if x.name in self.filter_ch else x)
		# Change from float64 to float 16
		#self.filtered = convertDfType(self.filtered)

	def iir_notch(self, signal2filt, notch_freq, quality_factor):
		# Design a notch filter using signal.iirnotch
		b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.fs)
		# Apply notch filter to the noisy signal using signal.filtfilt
		notch_filtered = signal.filtfilt(b_notch, a_notch, signal2filt)
		return notch_filtered

	# ---------------------------------------------------------------------------
	# ICA-related methods 
	# ---------------------------------------------------------------------------
	def applyFastICA(self, X, n_components=None, random_state=0):
		"""
		Aply fast ICA algorithm: a fast algorithm for Independent Component Analysis. 
		https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html?msclkid=d421974caff811ec913ff6f983941655
		
		Parameters
		------------
		X: array-like of shape (n_samples, n_record_ch). Training data, where n_samples is the number of samples and n_record_ch (n_features) is the number of channels or features.
		n_components: [int], default=None. Same as n_sources. Number of components to use. If None is passed, all are used.
		random_state: [int], RandomState instance or None, default=None. Used to initialize w_init when not specified, with a normal distribution. 
							Pass an int, for reproducible results across multiple function calls. See Glossary.

		Returns
		------------
		icaTransformer
		mixing_: 	[ndarray-like of shape (n_record_ch, n_sources)]. The pseudo-inverse of components_. It is the linear operator that maps independent sources to the data.
		sources_: 	[array-like of shape (n_samples, n_sources/n_components)]. Estimated sources obtained by transforming the data with the estimated unmixing matrix.
		ica_ch: 	[list] range with the length of the number of sources/components. To know the original sources.
		"""
		# compute ICA:  X = AxS   A: mixing matrix   S: source matrix
		self.icaTransformer = FastICA(n_components=n_components, random_state=random_state)   # random_state pass an int, for reproducible results across multiple function calls. 
		self.sources_ = self.icaTransformer.fit_transform(X)  # Get the estimated sources (fit the model and recover the sources from X)
		self.mixing_ = self.icaTransformer.mixing_  # Get estimated mixing matrix
		self.ica_ch = list(np.arange(0, n_components, 1))
		return (self.sources_, self.mixing_)

	def removeICA(self, keep_ch):
		"""
		Remove ICA channels that are not relevant

		Parameters
		------------
		keep_ch: [list] list with the sources/components to be kept. 

		Return
		-----------
		ica_ch: [list] list with the sources/components to be removed. 

		"""
		try:
			[self.ica_ch.remove(i) for i in keep_ch]
		except: 
			('ICA channels have already been removed. Type: self.ica_ch = list(np.arange(1, n_components, 1)) to load the list again')
		print('Removed ICA channels: %s'%self.ica_ch)
	
	def applyInverseFastICA(self, dur_recording=None):
		"""
		Compute inverse ICA:  recover X after removing the noisy components of the ica transform

		Parameters
		------------
		dur_recording: 	[int]. Number of samples for the reconstructed dataframe. If None, take only the head of the dataframe

		Other used parameters
		----------------------
		ica_ch: 	[list] list with the sources/components to be removed. 
		ch_loc: 	[list of int] list with electrodes locations corresponding to the selected intan channels 
		map_array:	[numpy array] 1D array with the corresponding intan channels for linear electrode device (1-31 electrodes): map_array[0] is intan channel corresponding to electrode 1

		Return
		-----------
		reconstructed_df: [dataframe] with the results of the inverse ICA tranform

		"""
		for i in self.ica_ch:
			self.icaTransformer.mixing_[:, i] = np.zeros(np.shape(self.mixing_)[1])
		reconstructed_ = self.icaTransformer.inverse_transform(self.sources_)
		print(reconstructed_.shape)

		# Store results in dataframe
		reconstructed_df = pd.DataFrame()
		
		for i,k in enumerate(self.ch_loc): 
			print(i)
			print(k)
			reconstructed_df['ch_%s'%int(record.map_array[k])] = reconstructed_[:, i] 
		
		# Set datetime index
		if dur_recording is None:
			reconstructed_df['seconds'] = np.asarray(self.recording['seconds'])
		else:
			reconstructed_df['seconds'] = np.asarray(self.recording['seconds'].head(dur_recording))
		reconstructed_df.index = pd.DatetimeIndex(reconstructed_df.seconds * 1e9)
		reconstructed_df.index.name = 'time'
		reconstructed_df.name = 'reconstructed_df'
		
		return reconstructed_df
	
	def plot_ica_channels(self, dur_select, n_components, ylim=None, save_figure=True):
		"""
		Plot found ICA channels 

		Parameters
		------------
		dur_select: 	[int]. Number of samples to plot
		n_components: 	[int] Number of components to use. 
		ylim: 			[list of 1x2] Set the y-limits of the current axes: first position minimum y limit and second position maximum limit. If None, automatic ylim by pyplot. 
		save_figure: 	[boolean]. Determine if the figure is saved or not. Default True.
		"""
		fig, ax = plt.subplots(round(n_components/2),2, figsize=(10,10), sharex=True)
		ax = ax.flatten()
		for i, ic in enumerate(self.sources_.T):
			ax[i].plot(np.arange(0, dur_select)/self.fs,ic[0: dur_select], label='ica_%s'%i)
			ax[i].legend(loc='lower left')
			if ylim is not None:
				ax[i].set_ylim(ylim)
			# Hide the right and top spines
			ax[i].spines['right'].set_visible(False)
			ax[i].spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax[i].yaxis.set_ticks_position('left')
			ax[i].xaxis.set_ticks_position('bottom')

		if save_figure:
			now = datetime.datetime.now()
			current_time = now.strftime("%d%m%Y_%H%M%S")
			plt.savefig('%s/figures/ica-zoom-%s.png' %(self.path, current_time), facecolor='w')

	# ---------------------------------------------------------------------------
	# Wavelet-related methods 
	# ---------------------------------------------------------------------------
	def wavelet_analysis(self, signal, neural_wavelet, other_wavelet, ch, dtformat='%M:%S.%f', ylim=[-20,20], show_plot=True, figsize=(15,10),verbose=0):
		"""
		Wavelet decomposition and plots.
		Note that despite the whole dataframe is passed, the algorithm only applies to the selected channels (filter_ch)

		TO DO: check that substraction works well

		Parameters
		------------
		signal:			[dataframe] Dataframe with the signals to decompose
		neural_wavelet:	[MyWavelet object] Contains the parameters for the decomposition of neural events
		other_wavelet:	[MyWavelet object] Contains the parameters for the decomposition of noise events
		ch:				[int] illustrative channel to plot
		dtformat: 		[datetime string]. How to plot the dataframe x-axis. Default: '%M:%S.%f'
		ylim: 			[list of 1x2] Set the y-limits of the current axes: first position minimum y limit and second position maximum limit. 
		show_plot: 		[boolean] Signal to Show plot. Default True
		figsize: 		(float, float). default: rcParams["figure.figsize"] (default: [6.4, 4.8]) Width, height in inches.
		verbose:		[int - 0,1] signal to display text information (default 1 - show text, 0 - don't show) 

		Results
		---------
		neural_wvl:			 [dataframe] Structure with the results of the neural wavelet decomposition 
		neural_wvl_denoised: [dataframe] Structure with the results of the neural wavelet decomposition only after applying multirresolution analysis using DWT AND thersholding the coefficients.
		other_wvl:			 [dataframe] Structure with the results of the noise wavelet decomposition 
		other_wvl_denoised:  [dataframe] Structure with the results of the noise wavelet decomposition only after applying multirresolution analysis using DWT AND thersholding the coefficients.
		substraction_wvl: 	 [dataframe] Structure with the results of the substraction of the noise signal from the neural signal 
		"""
		#-------------------------------------
		# Neural signal decomposition
		#-------------------------------------
		if verbose:
			print("----------------------")
			print(" Neural decomposition")
		neural_wvl = signal.apply(lambda x: wavelet_decomp(x,  self.fs, 
														type=neural_wavelet.wvl_type,
														verbose=verbose,
														**neural_wavelet.kwargs) 
											if x.name in self.filter_ch else x)
		if 'thres_type' not in neural_wavelet.kwargs:
			print('no dwt')
			neural_wvl_denoised = []
			pass
		else:
			print(neural_wavelet.kwargs)
			""" Adaptive noise removal: Thresholding detail coefficients"""
			neural_wvl_denoised = signal.apply(lambda x: DWT_with_denoising(x,  self.fs,
															verbose=verbose,
															**neural_wavelet.kwargs) 
														if x.name in self.filter_ch else x)
		#-------------------------------------
		# Noise signal decomposition
		#-------------------------------------
		if verbose:
			print("----------------------")
			print(" Noise source decomposition (e.g. cardiac)")
		other_wvl = signal.apply(lambda x: wavelet_decomp(x,  self.fs, 
														type=other_wavelet.wvl_type,
														verbose=verbose,
														**other_wavelet.kwargs) 
											if x.name in self.filter_ch else x)
		if 'thres_type' not in other_wavelet.kwargs:
			print('no noise signal hard dwt thresholding')
			other_wvl_denoised = []
			pass
		else:
			print(other_wavelet.kwargs)
			""" Adaptive noise removal: Thresholding detail coefficients"""
			other_wvl_denoised = signal.apply(lambda x: DWT_with_denoising(x,  self.fs,
															verbose=verbose,
															**other_wavelet.kwargs) 
														if x.name in self.filter_ch else x)
		#---------------------------------
		# Original minus cardac component
		#---------------------------------
		substraction_wvl = self.original.apply(lambda x: x-other_wvl[x.name]
												if x.name in self.filter_ch else x)
		#-------------------------
		# Plots
		#-------------------------
		if show_plot:
			if (self.num_rows*self.num_columns)>1:
				text_neural = 'Neural component after wavelet decomposition'
				self.plot_signal(neural_wvl, ch, self.map_array, self.num_rows, self.num_columns, 
							 channels=self.ch_loc, text_label='', text_title=text_neural, ylim=ylim, show_plot=False)
				text_cardiac = 'Cardiac component after wavelet decomposition'
				self.plot_signal(other_wvl, ch, self.map_array, self.num_rows, self.num_columns, 
							 channels=self.ch_loc, text_label='', text_title=text_cardiac, ylim=ylim, show_plot=False)

			# If only one channel: one single figure with 3 subplots (original, neural decomp and cardiac decomp)
			if (self.num_rows*self.num_columns)==1:
				fig, ax = plt.subplots(5,1,figsize=figsize, sharex=True)
				ax[0].plot(neural_wvl['ch_%s'%int(ch)], lw=0.5, label='Neural component')
				if len(neural_wvl_denoised)>0:
					ax[1].plot(neural_wvl_denoised['ch_%s'%int(ch)], lw=0.5, label='Neural decomposition with threshold')
				ax[2].plot(other_wvl['ch_%s'%int(ch)], lw=0.5, label='Cardiac component')
				if len(other_wvl_denoised)>0:
					ax[3].plot(other_wvl_denoised['ch_%s'%int(ch)], lw=0.5, label='Noise decomposition with threshold')
				ax[4].plot(substraction_wvl['ch_%s'%int(ch)], lw=0.5, label='Substraction (original-other_wvl')
				ax[0].legend(loc='lower right')
				ax[1].legend(loc='lower right')
				ax[2].legend(loc='lower right')
				ax[3].legend(loc='lower right')
				ax[4].legend(loc='lower right')
				ax[0].set_title('%s'%ch)
				fig.suptitle('Signal decomposition', fontsize=16, family='serif')
		
				for i in range(len(ax)):
					# Hide the right and top spines
					ax[i].spines['right'].set_visible(False)
					ax[i].spines['top'].set_visible(False)

					# Only show ticks on the left and bottom spines
					ax[i].yaxis.set_ticks_position('left')
					ax[i].xaxis.set_ticks_position('bottom')
					# Format axes
					ax[i].xaxis.set_major_formatter(md.DateFormatter(dtformat))
		return (neural_wvl, neural_wvl_denoised, other_wvl, other_wvl_denoised, substraction_wvl)
	
	def signal_normalization(self, neural_wvl, other_wvl, ch, ylim=[-30,30], figsize=(15, 10), dtformat='%M:%S', show_plot=True):
		"""
		Normalization of wavelet signals. Scale input vectors individually to unit norm (vector length).

		Parameters
		------------
		neural_wvl: [dataframe] The neural data to normalize
		other_wvl: 	[dataframe] The noise data to normalize
		ch:			[int] illustrative channel to plot
		ylim: 		[array of 1x2] Set ylim of plot: first position minimum y limit and second position maximum limit
		show_plot: 	[boolean] True if show

		Results
		---------
		neural_wvl_norm:	[dataframe] Normalized neural data
		other_wvl_norm:		[dataframe] Normalized noise data 
		substraction_norm: 	[dataframe] Normalized substracted data
		"""
		neural_wvl_norm = neural_wvl.apply(lambda x: preprocessing.normalize(x.values.reshape(1, -1)).flatten()
								if x.name in self.filter_ch else x)

		other_wvl_norm  = other_wvl.apply(lambda x: preprocessing.normalize(x.values.reshape(1, -1)).flatten()
								if x.name in self.filter_ch else x)

		substraction_norm  = neural_wvl_norm.apply(lambda x: x-other_wvl_norm[x.name]
								if x.name in self.filter_ch else x)

		if show_plot==True:
			text_title='Normalised signals substracted (neural - cardiac)'
			self.plot_signal(substraction_norm, ch, self.map_array, self.num_rows, self.num_columns, 
						 channels=self.ch_loc, text_title=text_title, ylim=ylim, figsize=figsize, dtformat=dtformat, show_plot=show_plot)

		return (neural_wvl_norm, other_wvl_norm, substraction_norm)

	# ---------------------------------------------------------------------------
	# Spike detection-related methods 
	# ---------------------------------------------------------------------------
	def pipeline_peak_extraction(self, ch, noise_signal, noise_idx, other_detectionn_config, spike_detection_config, consider_noise=True, verbose=False):
		"""
		General method to extract noise and neural spikes and extract waveforms.
		
		Parameters
		------------
		ch:					[int] channel to analyse
		noise_signal:		[dataframe] structure with the signal to be processed as noise
		other_detectionn_config:[dict] parameters specifying the length of window, height... for detection ad extraction of noise events
		spike_detection_config: [dict] parameters specifying the length of window, height... for detection ad extraction of neural events
		consider_noise:		[boolean] discard neural peaks that lie within window of noise spike or not
		verbose:			[int - 0,1] signal to display text information (default 1 - show text, 0 - don't show) 

		Returns
		------------
		noise_idx:			[array] array with indexes of noise peaks (locations) after filtering with conditions
		noise_ampl_vector:	[array, same length as noise_idx] amplitude of noise signal correspoding to noise_idx 
		spikes_idx:			[array] array with indexes of neural peaks (locations) after filtering with conditions
		waveforms:			[array, rows: number of neural peaks, columns: window with indexes around neural events] 
		spikes_vector_ampl: [array, same length as signal] amplitude of neural signal correspoding to spikes_vector_loc 
		spikes_vector_loc:	[array] numpy array with same size as signal with 1 where there's a spike and 0 otherwise
		index_first_edge:	[array] indexes of detected peaks (positive/negative/both) before alignment to max or min
		"""

		time_start = time.time()
		# ----------------------------------------------------
		# Identify noise from noise signal
		# -----------------------------------------------------
		"""
		DONE
			- Select cardiac event window that comprises all the peaks and then if the neural peak 
			lies within it don't consider it.
		"""
		
		artifact_idx = []

		if verbose:
			print("-------------------------------")
			print("Detecting noise events")
			#print("ONLY WORKS for constant bpm")
			print('noise_detection_config: %s' %print(other_detectionn_config))
		print(len(noise_signal))
		print(len(noise_idx))
		if len(noise_signal)>0 and len(noise_idx)>0:
			print('Loading peaks')
			# Extract the signal window that coresponds to the window around a noise peak
			# from the signal that is being used to identify neural peaks
			noise_idx, noise_event_window = self.get_noise_window(noise_signal[ch], noise_idx, other_detectionn_config['general'])
			noise_ampl_vector = get_spike_amplitude(noise_signal[ch], noise_idx)

		elif len(noise_signal)>0 and len(noise_idx)==0:
			compute_height = False
			noise_idx = []
			# AG 04/02/22
			if 'height' not in other_detectionn_config['general']['find_peaks_args']:
				# AG 03/12/21
				if other_detectionn_config['general']['C']>0:
					if verbose:
						print('Using C=%s to compute threhold of noise' %other_detectionn_config['general']['C'])
					other_detectionn_config['general']['find_peaks_args']['height'] = (other_detectionn_config['general']['C'] * np.nanmedian(np.abs(noise_signal[ch])/0.6745), )

				else:
					print('Using median to compute threhold of noise')
					other_detectionn_config['general']['find_peaks_args']['height'] = np.nanmedian(np.abs(noise_signal[ch])/0.6745)*np.sqrt(np.log(len(noise_signal[ch])))
				
				compute_height = True

			# Get all spikes from signal (using signal.find_peaks)
			noise_idx = get_spikes(noise_signal[ch], self.fs,
										**other_detectionn_config['general'],										neo=False,verbose=verbose)
			#noise_idx = [*artifact_idx, *hr_peaks]
			#noise_idx = noise_idx.sort()

			# Restart height (if not pre-defined) when running multiple channels so it's computed every time
			if compute_height:
				other_detectionn_config['general']['find_peaks_args'].pop('height')

			# Extract the signal window that coresponds to the window around a noise peak
			# from the signal that is being used to identify neural peaks
			noise_idx, noise_event_window = self.get_noise_window(noise_signal[ch], noise_idx, other_detectionn_config['general'])
			noise_ampl_vector = get_spike_amplitude(noise_signal[ch], noise_idx)
		#elif len(noise_signal)==0 and len(hr_peaks)>0:
			#noise_idx = hr_peaks
			# Extract the signal window that coresponds to the window around a HR peak
			# from the signal that is being used to identify neural peaks
			#noise_idx, noise_event_window = self.get_noise_window(noise_signal[ch], noise_idx, other_detectionn_config['general'])
			#noise_ampl_vector = get_spike_amplitude(noise_signal[ch], noise_idx)
		
		else:
			print('Noise signal is empty, no peaks detected')
			noise_idx = []
			noise_ampl_vector = []
			noise_event_window = []

		# ----------------------------------------------------
		# Identify neural spikes from ENG
		# -----------------------------------------------------
		if verbose:
			print("-------------------------------")
			print("Detecting neural events")
		if consider_noise:
			print('Will discard noise events')
			noise_peaks=noise_event_window
		else:
			noise_peaks=[]
		spikes_idx, numunique_peak_pos, num_noise_peaks, index_first_edge = \
			self.spike_detection(self.signal2analyse[ch], self.detect_method, self.thresh_type,  
		 						noise_peaks=noise_peaks, verbose=verbose, # options for noise_peaks: [], noise_event_window
		 						**spike_detection_config)
		
		waveforms, spikes_idx = self.get_waveforms(ch, spikes_idx, spike_detection_config['general']['spike_window'], spike_detection_config['general']['min_thr'], spike_detection_config['general']['half_width'])
		spikes_vector_ampl = get_spike_amplitude(self.signal2extract[ch], spikes_idx)
		spikes_vector_loc = self.get_spike_location(self.signal2extract[ch], spikes_idx)	
		
		print("pipeline_peak_extraction completed. Time elapsed: {} seconds".format(time.time()-time_start))
		return noise_idx, noise_ampl_vector, spikes_idx, waveforms, spikes_vector_ampl, spikes_vector_loc, index_first_edge

	def get_noise_window(self, signal, noise_idx, config):
		"""
		This function takes the noise peaks extracted from the noise signal, filter the ones that don't meet
		conditions (e.g. minimum separation and outliers) and selects the window around these peaks
		
		Parameters
		----------
		signal: signal from where the HR spikes are extracted 
		noise_idx: array with original indexes of noise peaks identified from noise signal (e.g. HR)
		config: dictionary with configuration of noise detection

		Returns
		---------
		noise_idx:	[array] array with indexes of noise peaks (locations) after filtering with conditions
		noise_event_window:	[array, rows: number of noise peaks, columns: window with indexes around noise events] 
		"""
		noise_event_window = []
		delete_idx = []
		old_idx = -config['window']
		for i, new_idx in enumerate(noise_idx):
			# If the new HR peak doesn't meet the minimum separation between beats (new and previous one)
			# given by config['window'] then discard it. 
			# This is automatically done when using get_spikes(), which calls the python-implemented 
			# method find_peaks(), but in case we're using a different method
			if new_idx < (old_idx + config['window']):
				print('deleting: %s' %new_idx)
				noise_idx = np.delete(noise_idx, np.where(noise_idx == new_idx)) 
				continue
			''' Check this code because it's not working properly 
			if config['cardiac']: # Only for removing cardiac noise
				print('cardiac')
				# If the value of the HR peak in the signal is an outlier, discard peak (artifact)
				if signal[new_idx] > (np.mean(signal[noise_idx])+1.5*scipy.stats.iqr(signal[noise_idx])): 
					print('outlier')
					delete_idx.append(i) 
				print('before conti')
				continue
			'''
			# Get noise window centered around peak
			old_idx = new_idx
			new_noise = np.arange(new_idx-config['spike_window'][0], new_idx+config['spike_window'][1], 1)
			new_noise = new_noise[new_noise > config['spike_window'][0]]  # To avoid first sample artifact
			new_noise = new_noise[new_noise < (len(signal) - config['spike_window'][1])]
			noise_event_window.extend(new_noise)
		noise_idx = np.delete(noise_idx, delete_idx)
	
		return noise_idx, noise_event_window
	'''
	def template_match(self, signal, noise_idx, config):
		"""
		NOT IN USE
		This function takes the HR peaks extracted from the cardiac signal and matches them 
		with the correspondent peaks in the extracting_peaks_signal ('signal'). Then selects 
		also the window around these peaks, the match in the signal 

		Parameters
		----------
		signal: signal from where the neural spikes will be extracted (extracting_peaks_signal)
		noise_idx: array with indexes of HR peaks identified from cardiac signal
		config: dictionary with configuration of cardiac window length
		"""
		noise_event_window = []
		max_peaks = []
		sum_peaks = []
		
		# First get the maximum of the signal in the cardiac window 
		old_idx = -config['window']
		for i, hr in enumerate(noise_idx):
			# If the new HR peak doesn't meet the minimum separation between beats (new and 
			# previous one)  given by config['window'] then discard it. 
			# This is automatically done when using get_spikes(), which calls the python-implemented 
			# method find_peaks(), but in case we're using a different method
			if hr < (old_idx + config['window']):
				noise_idx = np.delete(noise_idx, np.where(noise_idx == hr)) 
				pass
			else:
				# Get HR peak and the cardiac window centered around it
				old_idx = hr
				new_noise = np.arange(hr-config['spike_window'][0], hr+config['spike_window'][1], 1)
				new_noise = new_noise[new_noise > config['spike_window'][0]]  # To avoid first sample artifact
				new_noise = new_noise[new_noise < (len(signal) - config['spike_window'][1])]
				# Take the maximum value of the signal from each cardiac window
				max_peaks.append(signal[new_noise].max())
		 
		delete_idx = []
		for x, m in enumerate(max_peaks):
			# If the value of the HR peak in the signal is over normal range, discard peak (artifact)
			if m > (statistics.mean(max_peaks)+0.6*statistics.stdev(max_peaks)): 
				delete_idx.append(x) 
			else:
			# Get window around HR peaks in signal
				p = noise_idx[x]
				new_noise = np.arange(p-config['spike_window'][0], p+config['spike_window'][1], 1)
				new_noise = new_noise[new_noise > config['spike_window'][0]]  # To avoid first sample artifact
				new_noise = new_noise[new_noise < (len(signal) - config['spike_window'][1])]
				noise_event_window.extend(new_noise)		
		noise_idx = np.delete(noise_idx, delete_idx)
		
		"""
		fig, axes = plt.subplots(1,1)	
		axes.plot(signal[ch].to_numpy(), '-', linewidth=0.5, label='Signal')
		axes.scatter(np.array(list(noise_idx)), np.ones(len(noise_idx))*0.01, marker='o', label='spikes')
		"""
		return noise_idx, noise_event_window
	'''

	def spike_detection(self, signal, method, thresh_type, 
						manual_thres=None, noise_peaks=[], verbose=False, **kwargs):
		"""
		Method for identifying neural spikes based on different methods

		Parameters
		------------
		signal: 		self.signal2analyse[ch]
		method: 		[string] 
		thresh_type: 	[string] positive, negative or both
		manual_thres: 	[None or float] If not None, then select the value specified as manual_thres
		noise_peaks: 	[array, rows: number of noise peaks, columns: window with indexes around noise events] 
		verbose:		[int - 0,1] signal to display text information (default 1 - show text, 0 - don't show) 

		Returns
		-----------
		spikes_idx:			[array] array with indexes of neural peaks (locations)
		numunique_peak_pos: [int] final number of neural spikes
		num_noise_peaks:	[int] final number of noise spikes
		index_first_edge: 	[array] indexes of detected peaks (positive/negative/both) before alignment to max or min

		"""
		#------------------------------------
		# Method 1: get_spikes(). 
		# detect max of peaks using the find_peaks() method (using minimum distance between peaks)
		#--------------------------------------------
		''' The current implementation of get_spikes_method() significantly under-estimate the 
		number of negative peaks. I suspect that the function is not originally designed to handle
		negative peaks 
		'''

		if method == "get_spikes_method":
			if verbose:
				print(method)
			self.set_height(signal,  kwargs['general']['C'], self.manual_thres)
			print('Have set height')
			print(thresh_type)
			# get_spikes and get_spikes_simpleTh give same results when centering on max peak
			if thresh_type in ["positive", "both_thresh"]:
				print('get positive peaks')
				print(kwargs['general'])
				# Get positive peaks 
				#if self.threshold[0]>min_thr:
				'''
				if kwargs['general']['find_peaks_args']['height']: 
					print('height previously defined')
					print(kwargs['general']['find_peaks_args']['height'])
				else: 
					print('calculating height')
					print('Max height: %s' %kwargs['general']['Cmax'] * np.median(np.abs(signal)/0.6745))
					print('Min height: %s' %self.threshold[0])
					kwargs['general']['find_peaks_args']['height'] = (self.threshold[0], kwargs['general']['Cmax'] * np.median(np.abs(signal)/0.6745))
				'''
				print('calculating height')
				print('Max height: %s' %(kwargs['general']['Cmax'] * np.nanmedian(np.abs(signal)/0.6745)))
				print('Min height: %s' %self.threshold[0])
				kwargs['general']['find_peaks_args']['height'] = (self.threshold[0], kwargs['general']['Cmax'] * np.nanmedian(np.abs(signal)/0.6745))
				print('Height peaks range:')
				print( kwargs['general']['find_peaks_args']['height'])
				#else:
				#	kwargs['general']['find_peaks_args']['height'] = (min_thr, )
				indspos = get_spikes(signal, self.fs,
													**kwargs['general'],
													neo=False)
				if verbose:
					print('Length array spike pos from find_peaks(): %s' %len(indspos))

			'''
			When using negative threshold, one needs to be careful that by default of the function signal.find_peaks(), the first
			 entry of height is always the minimum, while the second entry is maximum.
			----Wilson
			'''			
			if thresh_type in ["negative", "both_thresh"]:
				# Get negative peaks
				#if self.threshold[1]<min_thr:
				kwargs['general']['find_peaks_args']['height'] = [-9999999,self.threshold[1]]  # Modify this line so that height is a 1x2 array---Wilson
				#else:  
				#	kwargs['general']['find_peaks_args']['height'] = -min_thr
				indsneg = get_spikes(signal, self.fs,
													**kwargs['general'],
													neo=False)
		#------------------------------------------
		# Method 2: get_spikes_simpleTh()
		# Detect first crossings of signal over a threshold calculated with the std of the noise
		# Only unique peaks
		#--------------------------------------------
		if method == "get_spikes_threshCrossing":
			''' Conclusion from Wilson analysis regarding get_spikes_threshCrossing:
			1. Unlike the get_spike_method, the get_spikes_threshCrossing method produced similar 
			result for positive and negative peaks.
			2. As expected, reverting the signal swap the number of positive and negative peaks. 
			However, small deviation does exist for some reasons.
			3. When detecting only positive peaks, get_spike_method and get_spikes_threshCrossing 
			produces similar number.
			'''
			if verbose:
				print(method)

			#self.set_height(signal, self.manual_thres)
			self.set_height(signal, kwargs['general']['C'], kwargs['manual_thres'])  # edited by wilson
			
			if thresh_type in ["positive", "both_thresh"]:
				if verbose:
					print(thresh_type)
				# Get positive peaks
				#if self.threshold[0]>min_thr:
				kwargs['general']['find_peaks_args']['height'] = (self.threshold[0], kwargs['general']['Cmax'] * np.median(np.abs(signal)/0.6745))
				#else:
				#	kwargs['general']['find_peaks_args']['height'] = (min_thr, )
				indspos = get_spikes_simpleTh(signal, self.fs, 
												**kwargs['general'],
												neo=False)
			if thresh_type in ["negative", "both_thresh"]:	
				# Get negative peaks
				#if self.threshold[1]<min_thr:
				kwargs['general']['find_peaks_args']['height'] = self.threshold[1]		
				#else:  
				#	kwargs['general']['find_peaks_args']['height'] = -min_thr
				indsneg = get_spikes_simpleTh(signal, self.fs, 
												**kwargs['general'],
												neo=False)
		#--------------------------------------------
		# Method 3: based on methods 4 by Zanos
		#--------------------------------------------
		if method == "so_cfar":
			if verbose:
				print(method)
			# make parameters a function of the sampling rate and spike duration
			num_std = kwargs['cfar']['nstd_cfar']		 # number of standard deviations for the threshold
			#wdur = 1501 / self.fs  # SO-CFAR window duration in seconds   10501
			#gdur = 10 / self.fs	# SO-CFAR guard duration in seconds   100
			wdur = kwargs['cfar']['wdur']
			gdur = kwargs['cfar']['gdur']

			w = int(round(wdur * self.fs))
			if (w % 2) == 0:
				w = w + 1
			g = int(round(gdur * self.fs))
			if (g % 2) == 0:
				g = g + 1
				#print(w)
				#print(g)

			indspos, indsneg, thresh = so_cfar_Zanos(signal, w, g, nstd= [num_std, num_std], verbose=verbose)

			# 17/01/2022 Remove spikes that are separated less than refractory distance. Only works for all positive or all negative peaks
			distance = kwargs['general']['find_peaks_args']['distance']
			#indspos2 = np.delete(indspos, np.where(numpy.diff(indspos)<distance)[0]+1) # Plu one because with diff we are one index behind
			try:
				# Positive peaks
				prev_p = indspos[0]
				dist_peaks_p = [prev_p]
				for p in indspos[1:]:
					if (p-prev_p) >= distance:
						dist_peaks_p.append(p)
						prev_p = p
					else:
						continue
				indspos = np.asarray(dist_peaks_p)
				# Negative peaks
				prev_n = indsneg[0]
				dist_peaks_n = [prev_n]
				for p in indsneg[1:]:
					if (p-prev_n) >= distance:
						dist_peaks_n.append(p)
						prev_n = p
					else:
						continue
				indsneg = np.asarray(dist_peaks_n)
			except:
				print('No peaks detected')
				
			# Save threshold in object
			self.threshold = [thresh[:,0].transpose(), thresh[:,1].transpose()]
			#self.threshold = np.append(self.threshold, thresh[:,0], axis=1)
			#self.threshold = np.append(self.threshold, thresh[:,1], axis=1)
			#print(self.threshold[0])
			if verbose:
				print('threshold[0].max() %s:' %self.threshold[0].max())
			#print(len(self.threshold[0]))
			#print(self.threshold)
		
		#----------------------------------------------------------
		# Determine positive/negative/both peaks and waveforms
		#----------------------------------------------------------
		print('Determining waveforms')
		num_noise_peaks=0
		spikes_idx=[]

		index_first_edge = []
		if thresh_type == "both_thresh":
			# Before Feb 2024, using combine_threshold_crossing in Zanos (although different implementation)
			# From Wilson research, using his function
			index_first_edge = combine_threshold(indspos, indsneg)
			if verbose:
				print('Number of index_first_peaks: %s' %len(index_first_edge))

		if thresh_type == "positive":
			index_first_edge = indspos
			if verbose:
				print('Number of index_first_peaks: %s' %len(index_first_edge))

		if thresh_type == "negative":
			index_first_edge = indsneg
			if verbose:
				print('Number of index_first_peaks: %s' %len(index_first_edge))

		if thresh_type == "positive":
			# Align peaks around maxima in window from the filtered signal
			# Center in windows in raw signal: This needs to be done as the wavelet 
			# decomposition may slightly change the waveform, so we ensure we're picking 
			# up the maximum of the original signal
			'''
			spikes_idx, numunique_peak_pos, num_noise_peaks = \
				max_centered_peaks_parallel(signal, index_first_edge, 
											 kwargs['general']['spike_window'],
											 noise_peaks=noise_peaks,
											 verbose=verbose)
			
			'''
			spikes_idx, numunique_peak_pos, num_noise_peaks = \
				max_centered_peaks_optimized(signal, index_first_edge, 
											 kwargs['general']['spike_window'],
											 noise_peaks=noise_peaks,
											 verbose=verbose)
			
		if thresh_type == "negative":
			spikes_idx, numunique_peak_pos, num_noise_peaks = \
				min_centered_peaks_optimized(signal, index_first_edge, 
											 kwargs['general']['spike_window'],
											 noise_peaks=noise_peaks,
											 verbose=verbose)

		if verbose:
			print('Num cardiac peaks: %s ' %(num_noise_peaks))

		'''
		# PLOT THRESHOLD
		# Create one figure per channel, otherwise can't see anything
		fig_th, axes_th = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
		fig_th.suptitle('threshold evolution', fontsize=16, family='serif')

		t_start = signal.index[0].timestamp()
		try:
			t_th = t_start + np.arange(0,len(self.threshold[1]))/self.fs
			th_df = pd.DataFrame(t_th, columns=['t'])
			th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
			axes_th.plot(th_df.index, np.array(self.threshold[1]), '-', linewidth=0.5, label='pos th', color='black')
		except TypeError:
			print('Cannot plot threshold')
		try:
			t_th = t_start + np.arange(0,len(self.threshold[0]))/self.fs
			th_df = pd.DataFrame(t_th, columns=['t'])
			th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
			axes_th.plot(th_df.index, np.array(self.threshold[0]), '-', linewidth=0.5, label='pos th', color='black')
		except TypeError:
			print('Cannot plot threshold') 

		# Format axes
		axes_th.set_xlabel('Time')
		axes_th.set_ylabel('Voltage [uV]')
		# Hide the right and top spines
		axes_th.spines['right'].set_visible(False)
		axes_th.spines['top'].set_visible(False)
		# Only show ticks on the left and bottom spines
		axes_th.yaxis.set_ticks_position('left')
		axes_th.xaxis.set_ticks_position('bottom')
		axes_th.xaxis.set_major_formatter(md.DateFormatter(dtformat))
			
		# Get current time for saving (avoid overwriting)
		now = datetime.datetime.now()
		current_time = now.strftime("%d%m%Y_%H%M%S")
		if save_figure=='png':
			fig_th.savefig('%s/figures/threshold_evolution-%s-%s.png' %(self.path, ch,current_time), facecolor='w')
			fig.savefig('%s/figures/identified_peaks-%s-%s.png' %(self.path, ch,current_time), facecolor='w')

		'''
		return spikes_idx, numunique_peak_pos, num_noise_peaks, index_first_edge

	def set_height(self, signal, C, manual_thres=None):
		"""
		The first element is always interpreted as the minimal and 
		the second, if supplied, as the maximal required height (threshold).

		Parameter
		----------
		signal: [] 
		C: used 
		manual_thres: [array of 1x2] first element is the positive and second element is the negative manual thresholds

		Returns
		------------
		threshold:	

		Comments
		----------
		Using C instead of np.sqrt(2*np.log(len(signal)), the latter is normally used for denoising wavelet
		components [Diedrich2003]. It's however kept commented in case I want to use it in the future
		"""

		threshold = []
		if manual_thres is None:
			print('NOT A MANUAL THRESHOLD')
			# divided by 0.6745, which is the 75th percentile of the standard normal distribution [15],
			std_noise = np.median(np.abs(signal)/0.6745) 
			threshold.append(std_noise*C)  #AG 02/12/2021
			threshold.append(-std_noise*C)	#AG 02/12/2021
			#print('np.sqrt(2*np.log(len(signal)): %s' %np.sqrt(2*np.log(len(signal))))
			print('std_noise: np.median(np.abs(signal)/0.6745): %s' %np.median(np.abs(signal)/0.6745))
			#print('np.mean(np.abs(signal)/0.6745): %s' %np.mean(np.abs(signal)/0.6745))
			#thr_number = (std_noise)*np.sqrt(2*np.log(len(signal)))
			#threshold.append((std_noise)*np.sqrt(2*np.log(len(signal))))   
			#threshold.append(-(std_noise)*np.sqrt(2*np.log(len(signal))))	
		else:
			threshold.append(manual_thres[0])
			threshold.append(manual_thres[1])
		self.threshold = threshold
	'''
	# Old non-optimised function (changed by the below on 07.02.2025)
	def get_waveforms(self, channel, spikes_idx, spike_window, min_thr, half_width):
		"""
		Method to filter detected peaks based on their amplitude and width (to discard peaks with shapes different to AP),
		and extract the window around the identified neural spikes.

		Note we're using signal2extract (generally the filtered ENG), which may be different to signal2analyse

		Parameters
		------------
		channel:	  [int] channel to extract waveforms from 
		spikes_idx:	  [array] array with indexes of neural peaks (locations) before filtering with conditions
		spike_window: [array 1x2] Length of window in samples: first position are samples before identified spike, second position after. From spike_detection_config['general']
		min_thr:	  [array 1x1] Minimum of amplitude of identified spikes. From spike_detection_config['general']
		half_width:	  [array 1x2] Min and max length in seconds from min to max of waveform. From spike_detection_config['general']

		Returns
		------------
		wdws:		[array, rows: number of neural peaks, columns: window with indexes around neural events] 
		spikes_idx:	[array] array with indexes of neural peaks (locations) before filtering with conditions

		"""
		# 07/12/2021
		# Changed to extract waveform from analysed signal, and not self.recording (filtered signal is very deformed). 
		# Feinstein also extracts from decomposed neural signal
		
		print("Getting waveforms")
		waveforms = []
		original_spikes = spikes_idx
		nowidth = 0
		noamp = 0
		index = 0

		while index in range(len(original_spikes)): 
			# Don't remove items from a list while iterating over it; iteration will skip items as the iteration index is not updated to account for elements removed.
			#Note that in a for loop the iteration variable cannot be changed. It will always be set to the next value in the iteration range. Therefore the problem is 
			# not the enumerate function but the for loop. Therefore use a while
			p = original_spikes[index]
			# First filter peaks based on amplitude
			# First filter peaks based on amplitude. ps: I add abs in min and max thr in the following line so that it works for negative thres----wilson
			if np.abs(self.signal2extract[channel][p])>np.abs(min_thr[0]) and np.abs(self.signal2extract[channel][p])<np.abs(min_thr[1]): # or self.signal2extract[channel][p]<min_thr[1]
				#if np.abs(self.signal2extract[channel][p])>min_thr[0] and np.abs(self.signal2extract[channel][p])<min_thr[1]: or self.signal2extract[channel][p]<min_thr[1]:
				try:
					new_window = np.arange(p - spike_window[0], p + spike_window[1] )
					# Second filter based on width
					# Calculate absolute distances from zero_crossings to max value and get the smallest of them. This is the closest zero crossing
					cross_zero = np.where(np.diff(np.sign(self.signal2extract[channel][new_window])))[0]+1
					dist = np.abs(np.argmax(self.signal2extract[channel][new_window])-cross_zero)
					min2max = np.sort(dist)[0]
					if min2max > int(self.fs*half_width[0]) and min2max < int(self.fs*half_width[1]):   # Half width longer than 1 sec
						waveforms.append(self.signal2extract[channel][new_window]) 	
						index = index + 1
					else: 
						spikes_idx.remove(p)
						nowidth = nowidth+1
						
				except IndexError:
						print('get_waveforms(): no zero-crossing within window ')
						spikes_idx.remove(p)
					#sys.exit()
			else: 
				noamp = noamp+1
				spikes_idx.remove(p)

		if waveforms:
			wdws = np.stack(np.array(waveforms))
		else:
			wdws = []
		print('Peaks excluded - not appropriate width: %s' %nowidth)
		print('Peaks excluded - not appropriate amplitude: %s' %noamp)
		print('Len filtered peaks: %s' %len(spikes_idx))
		print('Len wdsw peaks: %s' %len(wdws))
		return wdws, spikes_idx
	'''
	
	def get_waveforms(self, channel, spikes_idx, spike_window, min_thr, half_width):
		"""
		Extracts neural spike waveforms while filtering based on amplitude and width constraints. New optimised 07.02.2024
		"""
		print("Getting waveforms")
		waveforms = []
		original_spikes = spikes_idx
		valid_spikes = []  # To track valid spikes
		
		nowidth, noamp = 0, 0
		
		abs_min_thr = np.abs(min_thr)  # Precompute threshold values
		half_width_samples = (int(self.fs * half_width[0]), int(self.fs * half_width[1]))  # Convert to sample counts
		
		for p in original_spikes:
			amp = np.abs(self.signal2extract[channel][p])
			
			# Amplitude filter
			if abs_min_thr[0] <= amp <= abs_min_thr[1]:
				try:
					new_window = np.arange(p - spike_window[0], p + spike_window[1])
					signal_segment = self.signal2extract[channel][new_window]
					
					# Width filter based on zero crossings
					cross_zero = np.where(np.diff(np.sign(signal_segment)))[0] + 1
					if len(cross_zero) > 0:
						min2max = np.min(np.abs(np.argmax(signal_segment) - cross_zero))
						if half_width_samples[0] <= min2max <= half_width_samples[1]:
							waveforms.append(signal_segment)
							valid_spikes.append(p)
						else:
							nowidth += 1
					else:
						nowidth += 1
				except IndexError:
					print(f'get_waveforms(): No zero-crossing found within window at index {p}')
					nowidth += 1
			else:
				noamp += 1
		
		wdws = np.stack(waveforms) if waveforms else []
		spikes_idx = valid_spikes  # Set spikes_idx to valid spikes
		
		print(f'Peaks excluded - not appropriate width: {nowidth}')
		print(f'Peaks excluded - not appropriate amplitude: {noamp}')
		print(f'Len filtered peaks: {len(spikes_idx)}')
		print(f'Len wdsw peaks: {len(wdws)}')
		
		return wdws, spikes_idx

	"""
	def get_spike_amplitude(self, signal, spikes_idx=[]):   # Moved to utils as it's a general method also used in waveforms
		spikes_vector_ampl = np.zeros(len(signal))
		if type(spikes_idx) is 'list':
			spikes_idx_array = np.array(list(spikes_idx))
			sys.exit()
			if spikes_idx:
				spikes_vector_ampl[spikes_idx_array] = signal[spikes_idx_array]
			else:
				spikes_vector_ampl = []
		else: # It's an array
			if spikes_idx != []:
				spikes_vector_ampl[spikes_idx] = signal.iloc[spikes_idx]
			else:
				spikes_vector_ampl = []
		#self.spikes_amplitudes = spikes_vector_ampl  # Not used, commented on 14/01/22
		return spikes_vector_ampl   
	"""


	def get_spike_location(self, signal, spikes_idx):  
		"""
		Parameters:
		------------- 
			signal: input signal being analysed (self.signal2extract[ch])
			spikes_idx: [array] 1D vector containing indexes of locations where neural spikes were identified 
		Return: 
		-------------
			spikes_vector_loc: numpy array with same size as signal with 1 where there's a spike and 0 otherwise
		"""

		spikes_vector_loc = np.zeros(len(signal))
		if spikes_idx:
			spikes_vector_loc[np.array(list(spikes_idx))] = 1
		else:
			spikes_vector_loc = []
		self.spikes_locations = spikes_vector_loc   
		return spikes_vector_loc   

	
	# ---------------------------------------------------------------------------
	# Methods to compute absolute and rolling metrics from detected peaks
	# ---------------------------------------------------------------------------
	def compute_metrics_spikes(self, channel, spike_type, waveforms, mymetrics=['peak', 'auc'], verbose=False):
		"""
		Method to compute maximum amplitude, auc of peaks
		To do: only works for stimulation periods? Need to generalised it for clusters

		"""
		wave_class = [] #list(np.empty([1,max(spike_type)+1]))
		max_peaks = []
		auc = []
		row_results = []

		# Initialise
		if self.results is None:
			self.results = pd.DataFrame()
		if self.summary is None:
			self.summary = pd.DataFrame()

		for i in np.arange(max(spike_type)+1):
			wave_class.insert(i, [j for j in np.where(spike_type==i)[0]])
			max_peaks.insert(i, [max(w) for w in waveforms[wave_class[i]]])
			auc.insert(i, [metrics.auc(np.arange(len(waveforms[0])), w) for w in waveforms[wave_class[i]]])
			self.results['max_peaks_%s_code_%s' %(channel, i)] = pd.Series(np.asarray(max_peaks[i]))
			self.results['auc_%s_code_%s' %(channel, i)] = pd.Series(np.abs(np.asarray(auc[i])))
			
			row_results.extend([self.results['max_peaks_%s_code_%s' %(channel, i)].mean(),
								self.results['max_peaks_%s_code_%s' %(channel, i)].std(),
								self.results['auc_%s_code_%s' %(channel, i)].mean(),
			 					self.results['auc_%s_code_%s' %(channel, i)].std()])
		self.summary.loc[channel] = row_results

	def compute_rolling_metrics(self, signal_df, ax, ch, window=1, units='s', plot_inc_over_base=False, dtformat='%M:%S.%f', show_plot=False, time_marks=[], verbose=False):
		"""
		Method to compute and plot overall summary metrics of all detected peaks (spike rate and amplitude)

		Parameters
		------------
		window: integer. length of rolling window 
		units: string. units for the window

		Return
		--------------
		self.summary
		signal_df['metric_spikes_rate_%s' %ch]	TO DO: what is this exactly and how does the summary looks like

		"""

		print(window)
		# -------------------------
		# Compute metrics on spikes
		# -------------------------
		if verbose:
			print('Computing metrics...')


		# Compute spikes rate (num of spikes per window)
		if verbose:
			print('Computing spikes rate')
		signal_df['metric_spikes_rate_%s' %ch] = \
		signal_df['spikes_locations_%s' %ch].rolling(window='%s%s' % (int(window), units),
		   											 min_periods=int(self.fs * window)).sum()
	 
		# Compute mean of amplitude of spikes in window length
		# Leave min_periods to at least the length of the window observations so that it's at least 1Hz(e.g. 10s window, need 10 obbservations for compouting, otherwise
			# returns np.nan()). Better this way because if it only requires 1 observation, then we can't really see overall changes on clustering dissapearing
		if verbose:
			print('Computing metric_amplitude')
		# AG 
		signal_df['spikes_amplitudes_%s' %ch] = signal_df['spikes_amplitudes_%s' %ch].replace(0, np.nan)
		signal_df['metric_amplitude_%s' %ch] = \
		signal_df['spikes_amplitudes_%s' %ch].dropna().rolling(window='%s%s' % (int(window), units),
																min_periods=10).mean() # Before dropnan was min_periods=int(self.fs * window) b

		if show_plot:
			if plot_inc_over_base:
				base_sr = signal_df['metric_spikes_rate_%s' %ch].min()
				base_amp = signal_df['metric_amplitude_%s' %ch].min()
			else:
				base_sr = 0
				base_amp = 0
			ax[0].plot(signal_df['metric_spikes_rate_%s' %ch]-base_sr, lw=0.5, label='Signal: %s'%ch)
			ax[1].plot(signal_df['metric_amplitude_%s' %ch]-base_amp, marker='o', markersize=1)
			ax[0].set_title('Spike rate')
			ax[0].set_xlabel('Time')
			ax[0].set_ylabel('Spikes per %s%s' %(int(window), units))
			ax[1].set_title('Mean amplitude')
			ax[1].set_xlabel('Time')
			ax[1].set_ylabel('Voltage [uV]')

			if len(time_marks)>0:
				mm_df = pd.DataFrame()
				try:
					t_start = signal_df.index[0].timestamp()
					t_mm = time_marks
				except TypeError:
					print('Converting time_marks to arrays..')
					t_mm = time_marks
				mm_df = pd.DataFrame(t_mm, columns=['t'])
				mm_df.index = pd.DatetimeIndex(mm_df.t * 1e9)
				ax[0].scatter(mm_df.index, np.ones(len(time_marks))*signal_df['metric_spikes_rate_%s' %ch].max(), marker='o')

			for i in range(len(ax)):
				# Hide the right and top spines
				ax[i].spines['right'].set_visible(False)
				ax[i].spines['top'].set_visible(False)

				# Only show ticks on the left and bottom spines
				ax[i].yaxis.set_ticks_position('left')
				ax[i].xaxis.set_ticks_position('bottom')
				# Format axes
				ax[i].xaxis.set_major_formatter(md.DateFormatter(dtformat))

			ax[0].legend(loc='upper right')
			plt.show()
		else:
			print('Plot will not show')
			plt.close()
		print('-------------------------------')
		pd.set_option("display.max_rows", None, "display.max_columns", None)
		
		self.summary.loc['%s' % ch]  = [signal_df['metric_spikes_rate_%s' %ch].max(), 
										signal_df['metric_spikes_rate_%s' %ch].min(),
										signal_df['metric_amplitude_%s' %ch].max(),
										signal_df['metric_amplitude_%s' %ch].min()]
		summary_df = pd.merge(signal_df['metric_spikes_rate_%s' %ch].dropna(), signal_df['metric_amplitude_%s' %ch].dropna(), left_index=True, right_index=True)
		print(summary_df.tail())
		return summary_df

	#------------------------------------------
	# Plots 
	# Imported for unification but also implemented in visualization.plots)
	#----------------------------------------------
	# Define a custom function to format the tick labels as empty strings

	
	def plot_signal(self,signal, ch, num_rows,num_columns, channels, text_label, text_title, ylim=None, figsize=(10, 15),no_label=False, dtformat='%M:%S.%f', savefigpath='', show_plot=False):
		# Removed map_array and plot the intan channel from record.ch_loc (07/03/2024)
		fig, axes = plt.subplots(num_rows,num_columns, sharex=True, sharey=True, figsize=figsize)
		if (num_rows*num_columns)==1:
			axes.plot(signal['ch_%s'%int(ch)], label=text_label, lw=0.5, color='#aaa9a8')
			# Format axes
			axes.xaxis.set_major_formatter(md.DateFormatter(dtformat))
		else:
			axes = axes.flatten()
			for i,j in enumerate(channels):
				#print(j)
				try:
					print('plotting device ch %s' %j)
					axes[i].plot(signal['ch_%s'%int(j)], label='ch_%s'%int(j), lw=0.5, color='#aaa9a8')
					#axes[i].set_title('ch_%s'%int(map_array[j]))
					# Hide the right and top spines
					axes[i].spines['right'].set_visible(False)
					axes[i].spines['top'].set_visible(False)

					# Only show ticks on the left and bottom spines
					axes[i].yaxis.set_ticks_position('left')
					axes[i].xaxis.set_ticks_position('bottom')
					axes[i].legend(loc='upper right')
					# Format axes
					axes[i].xaxis.set_major_formatter(md.DateFormatter(dtformat))
					if no_label:
						axes[i].xaxis.set_major_formatter(FuncFormatter(hide_tick_labels))
						axes[i].yaxis.set_major_formatter(FuncFormatter(hide_tick_labels))
						axes[i].legend().set_visible(False)  # Comment this line to show the legend
						axes[i].set_title('')
				except KeyError:
					print('intan channel %s not found' %int(j))
		fig.suptitle(text_title, fontsize=16, family='serif')
		if ylim is not None: 
			plt.ylim(ylim)

		if savefigpath!='':
			plt.savefig(savefigpath, facecolor='w')

		if show_plot==True:
			plt.show()
		else:
			print('Plot will not show')
			plt.close()

	def plot_signal_bokeh(self, signal, ch, map_array, num_rows,num_columns, channels, text_label, text_title, ylim=None, figsize=[250, 250], dtformat='%M:%S.%f', savefigpath='', show_plot=False):
		plot_options = dict(width=figsize[0], plot_height=figsize[1], tools='pan,box_zoom,wheel_zoom,reset,save')
		axes = []
		count = 0
		for i,j in enumerate(channels):
			try:
				print('plotting ch %s' %map_array[j])
				p = figure(x_axis_type="datetime", title='Voltage [uV]', y_range=ylim, **plot_options)
				p.xaxis.formatter.minsec = dtformat
				p.xaxis.major_label_orientation = pi/3
				p.line(signal.index, signal['ch_%s'%int(map_array[j])])
				axes.append(p)
				count = count+1			
				if count == 3:
					show(gridplot([axes]))
					count = 0
					axes = []
			except KeyError:
				print('channel %s not found' %int(map_array[j]))

		show(gridplot([axes]))

		if savefigpath!='':
			plt.savefig(savefigpath, facecolor='w')

		if show_plot==True:
			plt.show()
		else:
			print('Plot will not show')
			plt.close()


	def plot_freq_content(self, signal2plot, ch, nperseg=512, max_freq=10000, ylim=None, dtformat='%M:%S.%f', figsize=(10, 15), savefigpath='', show=False, cmap='viridis',  colorbar_ticks=[], no_label=False):
		"""
		plt.specgram parameters: 
		NFFT : int
			The number of data points used in each block for the FFT. A power 2 is most efficient. The default value is 256.
			The benefit of a longer FFT is that you will have more frequency resolution. The number of FFT bins, the discrete 
			fequency interval of the transform will be N/2. So the frequency resolution of each bin will be the sample frequency Fs x 2/N.
		mode : {'default', 'psd', 'magnitude', 'angle', 'phase'}
			What sort of spectrum to use. Default is 'psd', which takes the power spectral density. 
			'magnitude' returns the magnitude spectrum. 'angle' returns the phase spectrum without unwrapping. 
			'phase' returns the phase spectrum with unwrapping.
		scale : {'default', 'linear', 'dB'}
			The scaling of the values in the spec. 'linear' is no scaling. 'dB' returns the values in dB scale. When mode is 'psd', 
			this is dB power (10 * log10). Otherwise this is dB amplitude (20 * log10). 'default' is 'dB' if mode is 'psd' or 'magnitude' 
			and 'linear' otherwise. This must be 'linear' if mode is 'angle' or 'phase'.
		"""
		# Raw signal
		fig, ax = plt.subplots(3, 1, figsize=figsize)
		#ax[0].plot(self.original.index, signal2plot['ch_%s'%ch], linewidth=0.5, zorder=0)
		ax[0].plot(signal2plot["time"], signal2plot['ch_%s'%ch], color='#aaa9a8', linewidth=0.5, zorder=0)
		ax[0].set_title('Sampling Frequency: {}Hz'.format(self.fs))
		ax[0].set_xlabel('Time [s]')
		ax[0].set_ylabel('Voltage [uV]')
		if ylim is not None:
			ax[0].set_ylim(ylim)

		# PSD (whole dataset ferquency distribution)
		f_data, Pxx_den_data = signal.welch(signal2plot['ch_%s'%ch], self.fs, nperseg=512) # nperseg
		# ax[1].psd(data[0:sf], NFFT=1024, Fs=sf)
		ax[1].semilogx(f_data, Pxx_den_data)
		ax[1].set_xlabel('Frequency [Hz]')
		ax[1].set_ylabel('PSD [V**2/Hz]')

		# Spectogram (frequency content vs time)
		

		plt.subplot(313)
		#powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal2plot['ch_%s'%ch], NFFT=nperseg, Fs=self.fs, mode='psd', scale='dB', cmap=cmap)
		# plt.specgram plots 10*np.log10(Pxx) instead of Pxx
		if len(colorbar_ticks)==0:
			powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal2plot['ch_%s'%ch], NFFT=nperseg, Fs=self.fs, mode='psd', scale='dB', cmap=cmap)
			# Extract vmin from powerSpectrum
			vmin = np.min(powerSpectrum)
			print(vmin)
			vmax = np.max(powerSpectrum)
			colorbar_ticks = [vmin, vmax]
		else:
			vmin=colorbar_ticks[0]
			vmax=colorbar_ticks[-1]
		powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal2plot['ch_%s'%ch], NFFT=nperseg, Fs=self.fs, mode='psd', scale='dB', cmap=cmap,  vmin=vmin, vmax=vmax)
		plt.ylabel('Spectogram \n Frequenct [Hz]')
		plt.xlabel('Time [s]')
		plt.ylim([0, max_freq])
		clb = plt.colorbar(imageAxis,ax=plt.gca(), ticks=colorbar_ticks, format='%.1f')
		clb.ax.set_title('10*np.log10 \n [dB/Hz]') 

		# Format axes
		for i in range(len(ax)):
			# Hide the right and top spines
			ax[i].spines['right'].set_visible(False)
			ax[i].spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax[i].yaxis.set_ticks_position('left')
			ax[i].xaxis.set_ticks_position('bottom')
			if no_label:
				#ax[i].set_xticks([])
				#ax[i].set_yticks([])
				ax[i].set_ylabel('')
				ax[i].set_xlabel('')
		ax[0].xaxis.set_major_formatter(md.DateFormatter(dtformat))

		if savefigpath!='':
			plt.savefig(savefigpath, facecolor='w')

		if show==True:
			plt.show()
		else:
			print('Plot will not show')
			plt.close()

	def compute_frequency_content(
		self,
		signal_1d: np.ndarray,
		fs: float,
		nperseg: int = 512,
		max_freq: float | None = None,
	):
		"""
		Compute frequency-domain representations of a 1D neural signal.

		Parameters
		----------
		signal_1d : np.ndarray
			Raw or filtered signal (1D).
		fs : float
			Sampling frequency in Hz.
		nperseg : int
			FFT window size.
		max_freq : float | None
			Optional frequency cutoff.

		Returns
		-------
		dict
			PSD and spectrogram components for GUI plotting.
		"""

		# --- PSD (Welch) ---
		psd_freqs, psd = signal.welch(
			signal_1d,
			fs=fs,
			nperseg=nperseg,
		)

		if max_freq is not None:
			mask = psd_freqs <= max_freq
			psd_freqs = psd_freqs[mask]
			psd = psd[mask]

		# --- Spectrogram ---
		spec_freqs, spec_times, spec_power = signal.spectrogram(
			signal_1d,
			fs=fs,
			nperseg=nperseg,
			scaling="density",
			mode="psd",
		)

		# Convert to dB (match matplotlib specgram behavior)
		spec_power_db = 10 * np.log10(spec_power + np.finfo(float).eps)

		if max_freq is not None:
			fmask = spec_freqs <= max_freq
			spec_freqs = spec_freqs[fmask]
			spec_power_db = spec_power_db[fmask, :]

		return {
			"psd_freqs": psd_freqs,
			"psd": psd,
			"spec_freqs": spec_freqs,
			"spec_times": spec_times,
			"spec_power_db": spec_power_db,
		}

	def plot_identified_peaks(self, signal, axes, spikes_idx,spikes_vector_ampl, neural_component, cardiac_signal, noise_idx, noise_ampl_vector,index_first_edge,num_rows, num_columns, dtformat='%M:%S.%f', verbose=False): 
		"""
		Method to plot all peaks independently of their cluster
		This method is no longer being used. Plots now with plot_identified_clusterpeaks()
		This is only used to test index_first_edge and if we use different extract and analyse signals
		"""
		# Create dataframe to plot the spikes (neural and cardiac) in datetime format. Need to create one for each becase they're different lengths 
		if verbose:
			print('Plotting identifies peaks')
		t_start = signal.index[0].timestamp()
		try:
			t_neural = t_start+spikes_idx/self.fs
			t_cardiac = t_start+noise_idx/self.fs
		except TypeError:
			print('Converting spikes_idx to arrays..')
			t_neural = t_start+np.asarray(spikes_idx)/self.fs
			t_cardiac = t_start+np.asarray(noise_idx)/self.fs
		spikes_df = pd.DataFrame(t_neural, columns=['t'])
		spikes_df.index = pd.DatetimeIndex(spikes_df.t * 1e9)
		hr_df = pd.DataFrame(t_cardiac, columns=['t'])
		hr_df.index = pd.DatetimeIndex(hr_df.t * 1e9)
		if len(index_first_edge)>0:
			t_index_first_edge = t_start+np.around(index_first_edge/self.fs, 6)
			index_first_edge_df = pd.DataFrame(t_index_first_edge, columns=['t'])
			index_first_edge_df.index = pd.DatetimeIndex(index_first_edge_df.t * 1e9)

		if verbose:
			print('Ploting identified peaks')
		if len(spikes_idx) != 0 and len(noise_idx) != 0:   
			axes.plot(signal, '-', linewidth=0.5, label='Signal')
			axes.scatter(spikes_df.index, spikes_vector_ampl[np.array(list(spikes_idx))], marker='o', label='Neural spikes')
			axes.plot(neural_component, '-', linewidth=0.5, label='Analysed signal') # Neural component
			axes.plot(cardiac_signal, '-', linewidth=0.5, label='Other components (cardiac / artifacts..)')
			try:
				#axes.scatter(hr_df.index, noise_ampl_vector[np.array(list(noise_idx))], marker='o', label='Other spikes (e.g. cardiac)') # 
				axes.scatter(hr_df.index, spikes_vector_ampl[np.array(list(noise_idx))], marker='o', label='Other spikes (e.g. cardiac)') # 
			except TypeError:
				print('noise spikes vector is empty, continue..')
			if len(index_first_edge)>0:
				axes.scatter(index_first_edge_df.index, np.ones(len(index_first_edge))*20, marker='o', label='initial spikes')
			try:
				t_th = t_start + np.arange(0,len(self.threshold[1]))/self.fs
				th_df = pd.DataFrame(t_th, columns=['t'])
				th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
				axes.plot(th_df.index, np.array(self.threshold[1]), '-', linewidth=0.5, label='neg th') 
			except TypeError:
				print('Cannot plot threshold')
			try:
				t_th = t_start + np.arange(0,len(self.threshold[1]))/self.fs
				th_df = pd.DataFrame(t_th, columns=['t'])
				th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
				axes.plot(th_df.index, np.array(self.threshold[0]), '-', linewidth=0.5, label='pos th')
			except TypeError:
				print('Cannot plot threshold')
			#axes.plot(record.signal_coded*500)
			# plt.legend(loc='upper right')   
			axes.set_title('%s' %signal.name)
		elif not spikes_idx:
			axes.plot(signal, '-', linewidth=0.5, label='Signal: %s' %signal.name)
			axes.plot(neural_component, '-', linewidth=0.5, label='Analysed signal')
			axes.plot(cardiac_signal, '-', linewidth=0.5, label='Other components (cardiac / artifacts..)')
			#axes.scatter(hr_df.index, noise_ampl_vector[np.array(list(noise_idx))], marker='o', label='Other spikes (e.g. cardiac)') # 
			axes.scatter(hr_df.index, spikes_vector_ampl[np.array(list(noise_idx))], marker='o', label='Other spikes (e.g. cardiac)') # 
			axes.title('%s' %signal.name, fontsize=14)
		elif not noise_idx:
			axes.plot(signal, '-', linewidth=0.5, label=' Signal: %s' %signal.name)
			axes.scatter(spikes_df.index, spikes_vector_ampl[np.array(list(spikes_idx))], marker='o', label='Neural spikes')
			axes.plot(neural_component, '-', linewidth=0.5, label='Analysed signal')
			axes.plot(cardiac_signal, '-', linewidth=0.5, label='Other components (cardiac / artifacts..)')
			axes.set_title('%s' %signal.name, fontsize=16)
			if len(index_first_edge)>0:
				axes.scatter(index_first_edge_df.index, np.ones(len(index_first_edge))*20, marker='o', label='initial spikes')
		else:
			pass
		# Format axes
		axes.set_xlabel('Time')
		axes.set_ylabel('Voltage [uV]')
		axes.legend(prop={"size":20})
		# Hide the right and top spines
		axes.spines['right'].set_visible(False)
		axes.spines['top'].set_visible(False)
		# Only show ticks on the left and bottom spines
		axes.yaxis.set_ticks_position('left')
		axes.xaxis.set_ticks_position('bottom')
		axes.xaxis.set_major_formatter(md.DateFormatter(dtformat))

	def compute_isi_distribution_over_time(
		self,
		spike_times_ms: np.ndarray,
		window_seconds: float = 10.0,
		isi_range_ms: int = 20,
		bin_size_ms: float = 1.0,
		perform_ks_test: bool = True,
	):
		"""
		Compute windowed ISI distribution matrix and optional Poisson KS test.

		Returns
		-------
		dict with:
			- isi_matrix (2D array)
			- window_centers_sec
			- isi_bins_ms
			- ks_stat (optional)
			- ks_p (optional)
		"""

		if len(spike_times_ms) < 2:
			raise ValueError("Not enough spikes to compute ISI.")

		window_period = window_seconds * 1000
		max_time = spike_times_ms[-1]

		windows = np.arange(0, max_time, window_period)

		isi_windows = []
		window_centers = []

		for start in windows:
			end = start + window_period
			mask = (spike_times_ms >= start) & (spike_times_ms < end)
			spikes_in_window = spike_times_ms[mask]

			if len(spikes_in_window) > 1:
				isi = np.diff(spikes_in_window)
				isi_windows.append(isi)
				window_centers.append(start + window_period / 2)

		# Histogram bins
		bins = np.arange(0, isi_range_ms + bin_size_ms, bin_size_ms)

		isi_matrix = np.zeros((len(isi_windows), len(bins) - 1))

		for i, isi in enumerate(isi_windows):
			hist, _ = np.histogram(isi, bins=bins)
			isi_matrix[i, :] = hist

		result = {
			"isi_matrix": isi_matrix,
			"window_centers_sec": np.array(window_centers) / 1000,
			"isi_bins_ms": bins[:-1],
		}

		# Optional KS test on full ISI distribution
		if perform_ks_test:
			full_isi = np.diff(spike_times_ms)

			lambda_rate = 1 / np.mean(full_isi)
			simulated_isi = expon.rvs(scale=1 / lambda_rate, size=len(full_isi))

			ks_stat, ks_p = ks_2samp(full_isi, simulated_isi)

			result["ks_stat"] = ks_stat
			result["ks_p"] = ks_p

		return result

	def gif(self, dataframe, topo_plot, samples, normalize=True, path='', make_contour=False, 
			plot_channels=True, plot_clabels=False, INTERP_POINTS=1000, show_plot=False, 
			make_gif=False, bar_title='Voltage [uV]'):

		"""
		dataframe: needs to have 'ch_x' as columns so that they can be filtered
		"""
		filenames = []
		
		if normalize:
			signal = dataframe.apply(lambda x: preprocessing.normalize(x.values.reshape(1, -1)).flatten()
						if x.name in self.filter_ch else x)
			data_array = signal[self.filter_ch].T.to_numpy()
			text = 'Normalised - '
		else:
			signal = dataframe
			data_array = signal[self.filter_ch].T.to_numpy()
			text = ""

		if topo_plot == "SC_topo":
			# Load locations file, specific for circular-structure locations
			Tk().withdraw() 
			if path=='':
				print("ERROR: Provide path for locations file")
				sys.exit()
			else:
				filepath = askopenfile(initialdir=path, title="Select 'locations' file (.csv)")
				locations = pd.read_csv(filepath)
		
		# Start running figures for creating the gif
		for sample in samples:
			if topo_plot == "SC_topo":
				# SC topo
				SC_topo_function(self.filter_ch, data_array[:,sample], 
									make_contour, locations, plot_channels, plot_clabels, INTERP_POINTS, bar_title)
				plt.title('%s SC Topograph at time %s ms' %(text,"{:.4f}".format((sample/self.fs)))) 
			elif topo_plot == "rectangular_topo":
				# Rectangular plot following the structure given by map_array
				plt.figure()
				z = np.zeros(len(self.map_array)) # For heatmap
				for n, j in enumerate(self.ch_loc):
					if np.isnan(self.map_array[j]):
						z[j] = np.nan
						pass
					else:
						# Get channel
						ch = 'ch_%s'%int(self.map_array[j])
						# Creating the array for heatmap
						z[j] = signal[ch].iloc[sample]
				z = np.reshape(z,(self.num_rows,self.num_columns))				
				""" 
				# Using matplot: more ugly
				z_matplot = np.flipud(z)
				plt.pcolormesh(z_matplot)
				clb = plt.colorbar()
				clb.ax.set_title('Amplitude/uV')
				"""
				# Using dataframe seaborn
				df = pd.DataFrame(z) #, index=Index, columns=Cols)
				sns.heatmap(df, annot=True)
				plt.title('%s Activation map at time %s ms' %(text,"{:.5f}".format(sample/int(self.fs)))) 								
				# Uncomment if show() [stops the run]
				if show_plot==True:
					plt.show()
				else:
					print('Plot will not show')
					plt.close()
			
			if make_gif:
				# create file name and append it to a list
				filename = 'gif_figures/%s.png' %sample
				filenames.append(filename)
				# save frame
				plt.savefig(filename)
				plt.close()

		if make_gif:
			# build gif
			with imageio.get_writer('%s/gif_activation.gif'%path, mode='I', loop=0) as writer:
				for filename in filenames:
					image = imageio.imread(filename)
					writer.append_data(image)
			# Remove files
			for filename in set(filenames):
				os.remove(filename)

	def bipolar_referencing_polars(self, signal: pl.DataFrame, filter_ch: list[str]) -> pl.DataFrame:
		"""
		Compute all-pairs bipolar referencing using Polars.

		Parameters
		----------
		signal : pl.DataFrame
			Filtered signal data (time × channels)
		filter_ch : list[str]
			List of channel column names (e.g. ['ch_4', 'ch_11', 'ch_20'])

		Returns
		-------
		pl.DataFrame
			Bipolar-referenced signals with columns 'ch_i-ch_j'
		"""

		# Ensure channel list is clean and ordered
		all_ch_list = [ch for ch in filter_ch if ch in signal.columns]

		if len(all_ch_list) < 2:
			raise ValueError("Bipolar referencing requires at least two channels")

		# Generate all possible bipolar pairs (same as itertools.combinations)
		bipolar_pairs = list(itertools.combinations(all_ch_list, 2))

		# Build Polars expressions for each bipolar pair
		bipolar_exprs = [
			(pl.col(ch1) - pl.col(ch2)).alias(f"{ch1}-{ch2}")
			for ch1, ch2 in bipolar_pairs
		]

		# Select only the bipolar-referenced signals
		references_df = signal.select(bipolar_exprs)

		print(f"Bipolar references computed: {references_df.shape}")
		print("Bipolar channels:", references_df.columns)

		return references_df

	def tripolar_referencing_polars(
		self,
		signal: pl.DataFrame,
		filter_ch: list[str],
		offset: int = 1,
	) -> pl.DataFrame:
		"""
		Compute tripolar referencing using Polars.

		Tripolar formula:
			2 * center - (proximal + distal)

		Parameters
		----------
		signal : pl.DataFrame
			Filtered signal (time × channels)
		filter_ch : list[str]
			Ordered list of channel names
		offset : int
			Neighbor distance (1 = immediate neighbors)

		Returns
		-------
		pl.DataFrame
			Tripolar-referenced signals
		"""

		if offset < 1:
			raise ValueError("Offset must be >= 1")

		if len(filter_ch) < 2 * offset + 1:
			raise ValueError(
				"Not enough channels for tripolar referencing "
				f"(need ≥ {2 * offset + 1})"
			)

		tripolar_exprs = []
		tripolar_labels = {}

		for i in range(offset, len(filter_ch) - offset):
			center = filter_ch[i]
			proximal = filter_ch[i - offset]
			distal = filter_ch[i + offset]

			col_name = f"{center}_tripolar_o{offset}"
			label = f"2×{center} - ({proximal} + {distal})"

			tripolar_exprs.append(
				(
					2 * pl.col(center)
					- (pl.col(proximal) + pl.col(distal))
				).alias(col_name)
			)

			tripolar_labels[col_name] = label

		references_df = signal.select(tripolar_exprs)

		# Attach labels as metadata (Polars equivalent of pandas .attrs)
		references_df = references_df.with_columns(
			[pl.lit(label).alias(f"{name}__label") for name, label in tripolar_labels.items()]
		).drop([c for c in references_df.columns if c.endswith("__label")])

		print(f"Tripolar referencing computed with offset = {offset}")
		print(tripolar_labels)

		return references_df

	def apply_referencing(self, method: str):
		"""
		Apply referencing to self.filtered and store result in self.referenced
		"""
		signal = self.filtered

		if method == "Median":
			all_ch_list = self.filter_ch
			print(all_ch_list)
			ref = signal.select(
				pl.concat_list(all_ch_list).list.median().alias("ref")
			)["ref"]

			self.referenced = signal.with_columns(
				[(pl.col(col) - ref).alias(col) for col in all_ch_list]
			)

		elif method == "Mean":
			all_ch_list = self.filter_ch

			ref = signal.select(
				pl.concat_list(all_ch_list).list.mean().alias("ref")
			)["ref"]

			self.referenced = signal.with_columns(
				[(pl.col(col) - ref).alias(col) for col in all_ch_list]
			)
		elif method == "Bipolar":
			self.referenced = self.bipolar_referencing_polars(
				self.filtered, self.filter_ch
			)
		elif method == "Tripolar":
			self.referenced = self.tripolar_referencing_polars(
				self.filtered, self.filter_ch, offset=1
			)
		elif method == "No Referencing":
			self.referenced = self.filtered
		else:
			raise NotImplementedError(f"Referencing method '{method}' not implemented")
	
	def detect_spikes_single_channel_polars(
		self,
		channel: str,
		height_std: float = 4.0,
		min_distance_ms: float = 3.0,
		):
			"""
			Detect spikes on a single channel using threshold + distance.
			Returns indices, times (s), and peak amplitudes.
			"""

			fs = self.fs
			signal = self.filtered[channel].to_numpy()

			# Threshold
			threshold = height_std * signal.std()

			# Minimum distance in samples
			min_distance_samples = int(min_distance_ms / 1000 * fs)

			# Detect peaks
			from scipy.signal import find_peaks
			indices, properties = find_peaks(
				signal,
				height=threshold,
				distance=min_distance_samples,
			)

			indices = pl.Series("indices", indices, dtype=pl.Int64)

			# Spike times (seconds)
			times = indices.cast(pl.Float64) / fs

			# Spike peak amplitudes
			peaks = pl.Series("peaks", signal[indices.to_numpy()])

			return {
				"indices": indices,
				"times": times,
				"peaks": peaks,
				"threshold": threshold,
			}
	
	def extract_spike_waveforms_polars(
        self,
        channel: str,
        spike_indices: np.ndarray,
        window_ms: float = 2.0,
        use_referenced: bool = True,
		) -> np.ndarray:
		"""
			Extract spike waveforms using Polars-backed signal.
		"""

		df = self.referenced if use_referenced and self.referenced is not None else self.filtered

		signal = df.select(channel).to_numpy().ravel()

		window = int(window_ms / 1000 * self.fs)
		waveforms = []

		for idx in spike_indices:
			start = max(0, idx - window)
			end = min(len(signal), idx + window)

			wf = signal[start:end]

			if len(wf) < 2 * window:
				pad_left = window - (idx - start)
				pad_right = window - (end - idx)
				wf = np.pad(wf, (pad_left, pad_right))

			waveforms.append(wf)

		return np.asarray(waveforms)
	
	def compute_average_waveform_polars(self, waveforms: np.ndarray):
		"""
		Compute mean and std waveform.
		"""

		wf_df = pl.DataFrame(waveforms)
		mean_wf = wf_df.mean().to_numpy().ravel()
		std_wf = wf_df.std().to_numpy().ravel()

		return mean_wf, std_wf
	
	def compute_spike_durations(
        self,
        waveforms: np.ndarray,
    ) -> np.ndarray:
		"""
		Compute spike durations in samples.
		"""

		durations = []

		for wf in waveforms:
			max_val = wf.max()
			th10 = 0.1 * max_val
			th90 = 0.9 * max_val

			duration = np.sum(wf > th10) - np.sum(wf > th90)
			durations.append(duration)

		return np.asarray(durations)
	
	def compute_isi_polars(self, spike_times_s: np.ndarray) -> pl.Series:
		"""
		Compute ISI in milliseconds using Polars.
		"""

		return (
			(pl.Series(spike_times_s)
			.diff() * 1000)
			.drop_nulls()
		)

	def single_channel_spike_analysis_polars(
        self,
        channel: str,
        height_std: float = 4.0,
        min_distance_ms: float = 3.0,
        extract_waveforms: bool = True,
    ):
		"""
		Complete single-channel spike analysis.
		"""

		result = self.detect_spikes_single_channel_polars(
			channel=channel,
			height_std=height_std,
			min_distance_ms=min_distance_ms,
		)

		if extract_waveforms:
			waveforms = self.extract_spike_waveforms_polars(
				channel=channel,
				spike_indices=result["indices"],
			)

			mean_wf, std_wf = self.compute_average_waveform_polars(waveforms)
			durations = self.compute_spike_durations(waveforms)
			isi = self.compute_isi_polars(result["times"])

			result.update({
				"waveforms": waveforms,
				"mean_waveform": mean_wf,
				"std_waveform": std_wf,
				"durations_samples": durations,
				"isi_ms": isi,
			})
		return {
			"title": f"Spike Analysis – {channel}",
			"plots": [
				{
					"title": "Detected Spikes",
					"x": result["times"],
					"y": result["peaks"],
					"kind": "scatter",
				},
				{
					"title": "Average Spike Waveform",
					"x": np.arange(len(mean_wf)),
					"y": mean_wf,
					"kind": "line",
				},
				{
					"title": "ISI Histogram",
					"x": None,
					"y": result["isi_ms"],
					"kind": "hist",
				},
			],
		}
class MyWaveforms:
	def __init__(self, waveforms, recording, fs, spikes_vector_loc, num_clusters, path):
		"""
		spikes_vector_loc: numpy array with same size as analysed signal with 1 where there's a spike and 0 otherwise
		"""
		self.waveforms = waveforms
		self.recording = recording
		self.fs = fs
		self.num_clusters = num_clusters
		self.spikes_vector_loc = spikes_vector_loc
		self.path = path

	#-------------------------------------------------------------------
	# Dimensionality reduction methods
	#-------------------------------------------------------------------
	# Standardization is a common pre-processing step before applying dimensionality reduction techniques
	# like UMAP (Uniform Manifold Approximation and Projection). The reason for standardizing the data is to 
	# ensure that all features are on the same scale and have zero mean.
	# Standardization is important because dimensionality reduction techniques like UMAP work based on distances
	# between points. If the features in the data have vastly different scales, then some features may dominate 
	# the distance calculation, leading to distorted or suboptimal results. By standardizing the data, we can ensure 
	# that all features contribute equally to the distance calculation and preserve the structure of the original data.
	# In practice, standardization can be achieved by subtracting the mean of each feature from each data point and 
	# dividing by the standard deviation of each feature. This ensures that each feature has a mean of zero and a standard deviation of one.
	
	def dim_red_pca(self, manual_comp_select=False, var_explained=0.9, pca_num_comp=0.99):
		time_start = time.time()

		## Apply min-max scaling on waveforms before applying PCA TO DO: Why? and which one to apply??
		# may be used when the upper and lower boundaries are well known 
		# from domain knowledge (e.g. pixel intensities that go from 0 to 255 in the RGB color range)
		scaler = sk.preprocessing.MinMaxScaler()
		dataset_scaled = scaler.fit_transform(self.waveforms)
		
		## OR standardizing the data: mean = 0 and scales the data to unit variance.
		dataset_scaled = StandardScaler().fit_transform(self.waveforms)
		
		## PCA
		if manual_comp_select:
			try:
				pca = PCA(n_components=pca_num_comp) # n_components=12
				self.pca_result = pca.fit_transform(self.waveforms)
			except TypeError:
				print('n_components too large for this dataset. Reducing to 3..')
				pca = PCA(n_components=pca_num_comp) # n_components=12
				self.pca_result = pca.fit_transform(self.waveforms)
		else:
			pca = PCA() 
			pca_result = pca.fit_transform(dataset_scaled)
			# Get optimal number of principal components (explained variance over )
			sumv = 0.
			n_components = 0
			while sumv<var_explained: 
				sumv += pca.explained_variance_ratio_[n_components]
				n_components+=1
			print('Num principal components: %s' %n_components)
			pca = PCA(n_components=n_components) 
			self.pca_result = pca.fit_transform(dataset_scaled)
		print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
	   
	def dim_red_tsne(self, n_components=2):
		time_start = time.time()
		## Standardizing the data: mean = 0 and scales the data to unit variance.
		dataset_scaled = StandardScaler().fit_transform(self.waveforms)
		## t-SNE (random_state=42 just for comparison purposes with UMAP)
		self.tsne_result = TSNE(random_state=42,n_components=n_components).fit_transform(dataset_scaled)
		print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	def dim_red_umap(self, n_neighbors=15, n_components=2):
		## Standardizing the data: mean = 0 and scales the data to unit variance.
		dataset_scaled = StandardScaler().fit_transform(self.waveforms)
		time_start = time.time()
		## UMAP
		try:
			reducer = umap.UMAP(random_state=42,n_neighbors=n_neighbors,n_components=n_components)
			self.umap_result = reducer.fit_transform(dataset_scaled)
		except TypeError:
			print("Reducing k...")
			try:
				reducer = umap.UMAP(random_state=42,n_neighbors=n_neighbors,n_components=n_components/2)
				self.umap_result = reducer.fit_transform(dataset_scaled)
			except TypeError:
				print('UMAP with n_components=3')
				try:
					reducer = umap.UMAP(random_state=42,n_neighbors=n_neighbors,n_components=3)  # Min number of components
					self.umap_result = reducer.fit_transform(dataset_scaled)
				except TypeError:
					print('Cannot compute UMAP.')
					self.umap_result = []
		print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))
	
	#-------------------------------------------------------------------
	# Clustering
	#-------------------------------------------------------------------


	def fit_WC3(self,data, spclustering, min_clus=20, elbow_min=0.4, c_ov=0.7, return_metadata=False):
			'''
			Super paramagnetic cluster 
			Removing self from spc.py (https://github.com/ferchaure/SPC) because the funcitons were not detected directly from the class
			Chaure FJ, Rey HG, Quian Quiroga R. A novel and fully automatic spike sorting implementation with variable number of features. J Neurophysiol , 2018. doi:10.1152/jn.00339.2018.
			'''
			classes, sizes= spclustering.run(data, return_sizes=True)

			maxdiff = np.max(np.diff(sizes[:,1:].astype(int),axis=0),1)
			maxdiff[maxdiff<0]=0

			main_cluster = sizes[:,0]

			prop = (main_cluster[1:]+maxdiff)/main_cluster[0:-1]
			aux = next((i for i in range(len(prop)) if prop[i]<elbow_min),np.NaN)+1 #percentaje of the rest

			# The next lines if removes the particular case where just a class is found at the 
			# lowest temperature and with just a small change the rest appears
			# all together at the next temperature
			if (not np.isnan(aux)) and spclustering.mintemp==0 and aux==2:
				aux = next((i for i in range(len(prop)-1) if prop[i+1]<elbow_min),np.NaN)+2 #percentaje of the rest
			tree = sizes[0:-1,:]
			clus = np.zeros_like(tree).astype(bool)

			clus[tree >= min_clus]=1; #only check the ones that cross the thr
			diffs =  np.diff(tree.astype(int),axis=0)
			clus = clus  * np.vstack([np.ones_like(clus[1,:]), diffs>min_clus])

			for ii in range(clus.shape[0]):
				detect = np.nonzero(clus[ii,:])[0]
				if len(detect>0):
					clus[ii,:detect[-1]]=1

			elbow = tree.shape[0]
			if not np.isnan(aux):
				clus[aux:,:] = 0
				elbow = aux


			if return_metadata:
				metadata = {'method': 'WC3'}
				allpeaks = np.where(clus)
				metadata['method_info'] = {'elbow':elbow, 'peaks_temp':allpeaks[0], 'peaks_cl':allpeaks[1]}

			for ti in reversed(range(elbow)):
				detect = np.where(clus[ti,:])[0]
				for dci in detect:
					cl = classes[ti,:] == dci
					for tj in np.arange(ti-1,-1,-1):
						toremove = np.where(clus[tj,:])[0]
						for rj in toremove:
							totest = classes[tj,:] == rj
							if sum(cl * totest)/min(sum(totest),sum(cl)) >= c_ov:
								clus[tj,rj]=0
			
			temp, clust_num = np.where(clus)
			c = 1 #initial class
			labels = np.zeros(classes.shape[1], dtype=int)
			for tx,cx in zip(temp, clust_num):
				labels[classes[tx,:]== cx] = c
				c += 1

			if return_metadata:
				metadata['clusters_info']={i+1: {'index':clust_num[i], 'itemp':ti} for i,ti in enumerate(temp)}
				metadata['sizes'] = sizes
				metadata['temperatures'] = spclustering.temp_vector.copy()
				return labels, metadata
			return labels

	def plot_temperature_plot(self,metadata,ax=None):
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		ax.set_yscale('log')
		ax.plot(metadata['temperatures'],metadata['sizes'])
		ax.set_prop_cycle(None)
		if metadata['method'] == 'WC3':
			ax.axvline(metadata['temperatures'][metadata['method_info']['elbow']],color='k',linestyle='--')
			ax.plot(metadata['temperatures'][metadata['method_info']['peaks_temp']],
				metadata['sizes'][metadata['method_info']['peaks_temp'],metadata['method_info']['peaks_cl']],color='k',alpha=0.5,marker='x',linestyle='',markersize=8)

		for c,info in metadata['clusters_info'].items():
			ax.plot(metadata['temperatures'][info['itemp']],
			metadata['sizes'][info['itemp'],info['index']],'.',markersize=15)
		ax.set_ylabel('Cluster Sizes')
		ax.set_xlabel('Temperatures')


	def clustering(self, ch,config, models, method='pca', roll_win=True, window=1, units='s', dtformat='%H:%M:%S', check_num_clusters=False, time_marks=[], show_plot=False,save_figure='png', verbose=False):
		"""
		cluster: vector of length equal to the number of peaks identified, with numbers corresponding to the cluster type.
		"""
		
		if method=='pca':
			data = self.pca_result
		elif method=='tsne':
			data = self.tsne_result
		elif method=='umap':
			data = self.umap_result
		elif method=='ica':
			data, _ = self.applyFastICA(self.waveforms, n_components=self.num_clusters, random_state=0)
		# -----------------
		# Clustering
		# -----------------
		if check_num_clusters:
			max_num_clusters = 50
			average_distance = clustering_avg_dis(data, max_num_clusters=max_num_clusters)

			# Plot elbow graph
			plot_check_cluster(average_distance, max_num_clusters=max_num_clusters)
			#sys.exit()

		# Assign waveforms to clusters
		for m in models:	
			cluster = []
			# define the model
			if m=='dbscan':
				time_start=time.time()
				model = DBSCAN(**config['dbscan'])
				# fit model and predict clusters
				cluster = model.fit_predict(data)
				cluster = cluster+1
				print ('DBSCAN done! Time elapsed: {} seconds'.format(time.time()-time_start))
			elif m=='spec':
				time_start=time.time()
				model = SpectralClustering(n_clusters=self.num_clusters)
				# fit model and predict clusters
				cluster = model.fit_predict(data)
				print ('SpectralClustering done! Time elapsed: {} seconds'.format(time.time()-time_start))
			elif m=='kmeans_AG':
				time_start=time.time()
				cluster, centers, distance = k_means(data, num_clus=self.num_clusters)
				print ('kmeans_AG done! Time elapsed: {} seconds'.format(time.time()-time_start))
			elif m=='superparamc':  #https://github.com/ferchaure/SPC
				spclustering = SPC(mintemp=config['spc']['mintemp'],maxtemp=config['spc']['maxtemp'])#,randomseed=randomseed)
				cluster, metadata = self.fit_WC3(data,spclustering,min_clus=config['spc']['min_clus'],return_metadata=True)
				#It is posible to show a temperature map using the optional output metadata
				self.plot_temperature_plot(metadata)
			else:
				if m=='kmeans':
					time_start=time.time()
					model = KMeans(n_clusters=self.num_clusters)
					print ('KMeans done! Time elapsed: {} seconds'.format(time.time()-time_start))
				if m=='mini_batch':
					time_start=time.time()
					model = MiniBatchKMeans(n_clusters=self.num_clusters)
					print ('MiniBatchKMeans done! Time elapsed: {} seconds'.format(time.time()-time_start))
				if m=='birch':
					time_start=time.time()
					model = Birch(n_clusters=self.num_clusters, **config['birch'])
					print ('Birch done! Time elapsed: {} seconds'.format(time.time()-time_start))
				if m=='gauss':
					time_start=time.time()
					model = GaussianMixture(n_components=self.num_clusters)
					print ('GaussianMixture done! Time elapsed: {} seconds'.format(time.time()-time_start))
				# fit the model
				model.fit(data)
				# assign a cluster to each example
				cluster = model.predict(data)
				print(cluster)
				print(len(cluster))
			
			# Retrieve unique clusters
			self.unique_clusters = np.unique(cluster)
			print('unique_clusters: %s' %self.unique_clusters)

			#plt.figure()
			for i in np.unique(cluster):
				cluster_mean = self.waveforms[cluster == i, :].mean(axis=0)
				print('max cluster %s mean: %s' %(i, np.max(cluster_mean)))
				#if c==0:
				#	plt.plot(*data[cluster==c,:].T,marker='.',color='k',linestyle='',markersize=3)
				#else:
				#	plt.plot(*data[cluster==c,:].T,marker='.',linestyle='',markersize=3)
				#plt.grid()
				#plt.title('Labeling')
				#plt.show()

			

			########################################################
			"""
			th_cluster = np.max(self.waveforms.mean(axis=0))/1.2
			cluster_spikes_vector_loc = self.spikes_vector_loc==1
			print('th_cluster: %s' %th_cluster)
			print('len(cluster): %s' %len(cluster))
			print('len(self.waveforms): %s' % len(self.waveforms))
			print('len(spikes_vector_loc): %s' % len(spikes_vector_loc))
			
			# Filter clusters based on amplitude
			for i in np.unique(cluster):
				print(i)
				print(len(self.cluster == i))
				print(len(self.waveforms))
				matches = np.where(cluster==i)[0]
				print('matches %s' %len(matches))
				#cluster_mean_array = [x for loc,x in enumerate(self.waveforms) if loc in matches]
				#cluster_mean = .mean(axis=0) #
				cluster_mean = self.waveforms[self.cluster == i, :].mean(axis=0)
				#print('cluster_mean: %s' %cluster_mean)
				if np.max(cluster_mean) < th_cluster:
					cluster = [x for x in cluster.tolist() if x!=i]
					cluster_spikes_vector_loc = [x for loc,x in enumerate(spikes_vector_loc) if loc not in matches]
					#self.waveforms = [x for loc,x in enumerate(self.waveforms) if loc not in matches]
			self.unique_clusters = np.unique(cluster)
			self.num_clusters = len(unique_clusters)
			print('len(cluster): %s' %len(cluster))
			print('len(self.waveforms): %s' % len(self.waveforms))
			print('len(spikes_vector_loc): %s' % len(spikes_vector_loc))
			print('unique_clusters after aplitude filtering: %s' %unique_clusters)		
			"""	
			########################################################

			# Store clusters
			self.cluster = cluster
			self.clustering_method = m

			if roll_win:
				self.cluster_dataframe = self.cluster_rolling_window(ch, window=window, units=units, verbose=verbose)

			# check the isi is not of noise waveforms
			#self.discard_isi_cluster()
			#print('unique_clusters after isi discard: %s' %self.unique_clusters)

			self.plot_cluster(data, ch, cluster, roll_win, window, units,dtformat, show_plot=show_plot, save=save_figure, time_marks=time_marks)


	def cluster_rolling_window(self, ch, window,units='s', show_plot=False, verbose=False):
		"""
		window: int
		"""

		# Compute rolling windiw
		#-----------------------------
		nonzero_cluster = [z+1 for z in self.cluster] # plus one to differenciate from 0 in cluster_vector (which means no spike)
		cluster_vector = self.get_cluster_vector(self.spikes_vector_loc, nonzero_cluster) 
		cluster_dataframe = pd.DataFrame()
		cluster_dataframe.index = pd.DatetimeIndex(self.recording.seconds * 1e9) # self.recording in wavelet is self.original
		cluster_dataframe.index.name = 'time'
		
		for c in self.unique_clusters: # Containes 0
			# Store in dataframe where waveforms of each cluster appear and the amplitude of the waveforms corresponding to each cluster type 
			cluster_dataframe['cluster_%s' %c] = cluster_vector==c+1 # plus one to differenciate from 0 in cluster_vector
			cluster_dataframe['amplitude_c%s'%c] = get_spike_amplitude(self.recording[ch], np.where(cluster_dataframe['cluster_%s'%c]==1)) #When there's a spike (True or 1) in that cluster type 


		# Get metrics in cluster 
			# Get number of elements in each cluster
		unique_elements, counts_elements = np.unique(self.cluster, return_counts=True)

		print('Number of element in each cluster: %s' % counts_elements)

		# print(len(neural) / fs)
		elem_sec = counts_elements / (len(self.recording) / self.fs)  # self.recording in wavelet is self.original

		if verbose:
			print('Number of element in each cluster per second: %s' % elem_sec)
			print('Number of element in each cluster per minute: %s' % (elem_sec * 60))

		# Compute metrics on clusters
		for c in self.unique_clusters:			# rc stands from 'rolling cluster' and c from 'cluster'
			# Spikes per window
			cluster_dataframe['rc_%s' %c] = cluster_dataframe['cluster_%s' %c].rolling(window='%s%s' % (int(window), units),
					  																	min_periods=int(self.fs * window)).sum()  
			# Spikes per second
			cluster_dataframe['src_%s' %c] = cluster_dataframe['cluster_%s' %c].rolling(window='%s%s' % (1, units),
					  																	min_periods=int(self.fs)).sum()  
			# Percentage of change with respect to previous second
			cluster_dataframe['pct_rc_%s'%c] = cluster_dataframe['rc_%s'%c].pct_change()*100  # periods=int(self.fs) periods mark the change with respect to which value, in this case respect 1 sec before
			cluster_dataframe['change_rc_%s'%c] = np.gradient(cluster_dataframe['rc_%s'%c])

			# Mean of amplitudes metric
			# AG 21/03/2022: TO avoid computation of 0s, substitute by nan and then drop them. 
			# min_interval needs to be reduced to windowsize (=1) to guarantee there's at least one spike
			# Leave min_periods to at least the length of the window observations so that it's at least 1Hz(e.g. 10s window, need 10 obbservations for compouting, otherwise
			# returns np.nan()). Better this way because if it only requires 1 observation, then we can't really see overall changes on clustering dissapearing
			cluster_dataframe['amplitude_c%s'%c] = cluster_dataframe['amplitude_c%s'%c].replace(0, np.nan)
			cluster_dataframe['metric_amplitude_c%s' %c] = cluster_dataframe['amplitude_c%s'%c].dropna().rolling(window='%s%s' % (int(window), units),
																										min_periods=10).mean()  # min_periods=int(self.fs * window)
			# For plotting purposes, substitute nan by 0
			cluster_dataframe['metric_amplitude_c%s' %c] = cluster_dataframe['metric_amplitude_c%s' %c].replace(np.nan, 0)

			# Percentage of change with respect to previous period
			cluster_dataframe['pct_amp_%s'%c] = cluster_dataframe['metric_amplitude_c%s'%c].pct_change()*100  # periods=int(self.fs) periods mark the change with respect to which value, in this case respect 1 sec before
			cluster_dataframe['change_amp_%s'%c] = np.gradient(cluster_dataframe['metric_amplitude_c%s'%c])
		return cluster_dataframe

	def discard_isi_cluster(self):
		# Unify colors of three plots:
		NUM_COLORS = len(self.unique_clusters)
		colors=[]
		cm = pylab.get_cmap('tab20c')  #https://matplotlib.org/stable/tutorials/colors/colormaps.html
		for i in range(NUM_COLORS):
			colors.append(cm(1.*i/NUM_COLORS)) # color will now be an RGBA tuple
		#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Default color list in python
		self.colors = colors
		print(colors)
		print()
		#colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

		print('Discarding clusters based on ISI')
		fig_a, axes_a = plt.subplots(math.ceil(len(self.unique_clusters)/(math.ceil(len(self.unique_clusters)/3))), math.ceil(len(self.unique_clusters)/3), figsize=(15, 8), sharex=True)
		fig_a.suptitle('ISI', fontsize=16, family='serif')  
		axes_a = axes_a.flatten()
		self.cluster_dataframe['seconds'] = self.recording.seconds
		original_clusters = self.unique_clusters
		for i,c in enumerate(original_clusters):
			s = self.cluster_dataframe['cluster_%s' %c].astype(int)
			list_pos = self.cluster_dataframe.seconds[s==1]
			max_isi = self.inter_spike_interv(list_pos, axes_a[c], color=colors[c],  bins=30, myrange=[0,30])
			print(max_isi)
			if max_isi < 7:
				self.unique_clusters = np.delete(original_clusters,i)
				print('Removing cluster %s, peak ISI at %s' %(c, max_isi))

	'''
	def get_cluster_vector(self, spikes_vector_loc=[], cluster=[]):
		"""
		Parameters:
		-------------
			spikes_vector_loc: numpy array with same size as signal with 1 where there's a spike and 0 otherwise
			cluster: vector of length equal to the number of peaks identified, with numbers corresponding to the cluster type (+1 to avoid non-spike).

		Return:
		-----------
			cluster_vector: numpy vector with same size as signal with the number of the type of cluster on the locations 
							where peaks were identified (given by spikes_vector_loc), and 0 otherwise.
		"""
		spikes_vector_loc = np.asarray(spikes_vector_loc)
		cluster = np.asarray(cluster)
		cluster_vector = np.zeros(len(spikes_vector_loc))
		if spikes_vector_loc.shape[0] > 0 and cluster.shape[0] > 0:
			cluster_vector[spikes_vector_loc==1] = cluster
		else:
			cluster_vector = []
		
		return cluster_vector   
	'''
	def get_cluster_vector(self, spikes_vector_loc=[], cluster=[]):
		"""
		Parameters:
		-------------
			spikes_vector_loc: numpy array with same size as signal with 1 where there's a spike and 0 otherwise
			cluster: vector of length equal to the number of peaks identified, with numbers corresponding to the cluster type (+1 to avoid non-spike).
	
		Return:
		-----------
			cluster_vector: numpy vector with same size as signal with the number of the type of cluster on the locations 
							where peaks were identified (given by spikes_vector_loc), and 0 otherwise.
		"""
		spikes_vector_loc = np.asarray(spikes_vector_loc)
		cluster = np.asarray(cluster)
	
		# Ensure input arrays are valid
		if spikes_vector_loc.ndim != 1 or cluster.ndim != 1:
			raise ValueError("Both inputs must be 1D numpy arrays.")
	
		cluster_vector = np.zeros_like(spikes_vector_loc, dtype=int)
	
		if spikes_vector_loc.shape[0] > 0 and cluster.shape[0] > 0:
			if np.sum(spikes_vector_loc == 1) != cluster.shape[0]:
				raise ValueError("Mismatch: Number of spikes in 'spikes_vector_loc' does not match 'cluster' length.")
	
			cluster_vector[spikes_vector_loc == 1] = cluster
	
		return cluster_vector

	#-------------------------------------------------------------------
	# Computing metrics from extracted waveforms
	#-------------------------------------------------------------------
	def snr(self, signal, ch, noise_samples=[0, 1000]):
		spk_avg = np.mean(self.waveforms, 0) 	# Average waveform
		spk_std = np.std(self.waveforms, 0) 	# Standard waveform
		spk_Vpp = np.max(spk_avg)-np.min(spk_avg)
		noise_sd = np.std(signal.iloc[int(noise_samples[0]):int(noise_samples[1])],0) #SD of noise
		print('sd noise: %s' %noise_sd)
		s2nr = spk_Vpp/(2*noise_sd)
		print('SNR: %s' %s2nr)
		print('Mean waveform Vpp: %s' %spk_Vpp)
		print('Need to have snrInfo declared in main code. See bran microwires to copy the needed bit')
		self.snrInfo.append('sd noise ch %s: %s	 ' %(ch, noise_sd))
		self.snrInfo.append('SNR ch %s: %s		  ' %(ch,s2nr))
		self.snrInfo.append('Mean waveform Vpp ch %s: %s' %(ch,spk_Vpp))
		return s2nr

	def inter_spike_interv(self,spikes_time, axes,color='g',bins=50, myrange=[0,60]):
		"""
		bins: number of equal-width bins in the range

		Suggestion: use same bins and range so that the max has a precision of 1ms
		"""
		isi = numpy.diff(spikes_time)
		# the histogram of the isi (by multiplying by 1000 we convert sec to msec)
		n, bins, patches = axes.hist(isi*1000, bins, range=myrange, density=False, facecolor=color, alpha=0.75) # bins: The edges of the bins. 
		max_peak_loc = bins[np.where(n==np.max(n))[0]] # Take the edge of the bin where the maximum value occurs

		axes.set_xlabel('ISI (msec)')
		axes.set_ylabel('Count')
		axes.text(10, np.max(n), ' Peak at %s msec' %max_peak_loc[0])

		# Hide the right and top spines
		axes.spines['right'].set_visible(False)
		axes.spines['top'].set_visible(False)

		# Only show ticks on the left and bottom spines
		axes.yaxis.set_ticks_position('left')
		axes.xaxis.set_ticks_position('bottom')
		return max_peak_loc

	#-------------------------------------------------------------------
	# Plots
	#-------------------------------------------------------------------
	def plot_waveforms(self, n=100):
		"""
		n: number of waverforms to plot
		"""
		print('Ploting waveforms')
		np.random.seed(10)
		fig, ax = plt.subplots(2,1,figsize=(10, 5))
		spike = []

		for i in range(n):
			spk = np.random.randint(0, self.waveforms.shape[0])
			ax[0].plot(self.waveforms[spk, :])
			spike.append(self.waveforms[spk, :])
		#print(spike)
		#print(np.sum(spike,1)/len(spike))
		meanWave = np.mean(spike,0)
		ax[1].plot(meanWave)

		# ax.set_xlim([0, 90])
		ax[0].set_ylabel('Voltage [uV]')
		ax[0].set_title('spike waveforms')
		ax[1].set_ylabel('Voltage [uV]')
		ax[1].set_title('mean spike waveforms')
		ax[1].set_xlabel('# sample')

		for i in range(len(ax)):
			# Hide the right and top spines
			ax[i].spines['right'].set_visible(False)
			ax[i].spines['top'].set_visible(False)

			# Only show ticks on the left and bottom spines
			ax[i].yaxis.set_ticks_position('left')
			ax[i].xaxis.set_ticks_position('bottom')

	def plot_waveforms_multipleCh(self, axes, spike_type, colors, code_names, ylim=[]):
		"""
		
		"""
		'Plotting waveforms'
		meanWave = np.mean(self.waveforms,0)
		axes.plot(meanWave, '--')

		for n_w, w in enumerate(self.waveforms):
			# Check that the spike is contained in the stimulation region and not
			# in the time2onset region (NaNs)
			if spike_type[n_w]<0:
				print('Plot_waveforms_multipleCh()')
				#print('Skipping spike (spike in time2onset region):' + str(n_w))
				continue
			else:
				axes.plot(w, c=colors[spike_type[n_w]], label=code_names[spike_type[n_w]], linewidth=0.5)
		
		if len(ylim)==0:
			pass	
		else:	
			axes.set_ylim(ylim)
		# Hide the right and top spines
		axes.spines['right'].set_visible(False)
		axes.spines['top'].set_visible(False)

		# Only show ticks on the left and bottom spines
		axes.yaxis.set_ticks_position('left')
		axes.xaxis.set_ticks_position('bottom')

		# Format axes
		axes.xaxis.set_major_formatter(md.DateFormatter('%M:%S.%f'))

	def plot_waveforms_multipleCh_mean(self, axes, ch, spike_window, ylim=[], dtformat='%S.%f'):
		"""
		
		"""
		print('Plotting waveforms')

		time = np.linspace(0, self.waveforms.shape[1] / self.fs, self.waveforms.shape[1]) * 1000
		#np.linspace(-spike_window[0]/self.fs,spike_window[1]/self.fs

		self.meanWave = np.mean(self.waveforms,0)
		std_waveforms = np.std(self.waveforms, 0)

		spike_window = spike_window*1000
		axes.plot(time, self.meanWave, '--', )
		axes.fill_between(time,
						 	self.meanWave - std_waveforms,
						 	self.meanWave + std_waveforms,
						 	alpha=0.15, label='Channel: %s'%ch)
		axes.legend()
		if len(ylim)==0:
			pass	
		else:	
			axes.set_ylim(ylim)
		# Hide the right and top spines
		axes.spines['right'].set_visible(False)
		axes.spines['top'].set_visible(False)

		# Only show ticks on the left and bottom spines
		axes.yaxis.set_ticks_position('left')
		axes.xaxis.set_ticks_position('bottom')

		

	def plot_identified_clusterpeaks(self, ch, signal, threshold, spikes_idx, noise_signal, noise_idx, dtformat='%M:%S.%f', ylim=[-100, 100],plot_legend=True, save_figure='png', verbose=False): 
		"""
		Optimised plot of detected spikes based on clusters (old method plot_identified_peaks())
		- Plot location of each peak corresponding to different cluster types
		- Remove signal2extract and signal2analyse (only one input signal) 
		"""
		if verbose:
			print('Plotting identified peaks')

		# Create one figure per channel, otherwise can't see anything
		fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
		fig.suptitle('Identified peaks', fontsize=16, family='serif')

		# Create one figure per channel, otherwise can't see anything
		fig_th, axes_th = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
		fig_th.suptitle('threshold evolution', fontsize=16, family='serif')

		t_start = signal.index[0].timestamp()

		if len(spikes_idx) != 0:   
			#axes.plot(signal, '-', linewidth=0.5, label='Signal')
			for c in self.unique_clusters: 
				cluster_signal = self.cluster_dataframe['cluster_%s' %c]
				#axes.scatter(signal[np.array(cluster_signal==1)].index, signal[np.array(cluster_signal==1)], marker='o', label='%s'%c, color=self.colors[c]) #label='%s'%c,, self.colors[c]
				axes.scatter(signal[np.array(cluster_signal==1)].index, signal[np.array(cluster_signal==1)]/signal[np.array(cluster_signal==1)]*(50+10*c), marker='o', label='%s'%c, color=self.colors[c]) #label='%s'%c,, self.colors[c]
			axes.plot(signal, '-', linewidth=0.5, label='Analysed signal') # Neural component
			if len(noise_signal)>0:
				axes.plot(noise_signal, '-', linewidth=0.5, color='black', label='Other components (cardiac, noise artifacts..)')
			try:
				axes.scatter(signal[np.array(list(noise_idx))].index, signal[np.array(list(noise_idx))]/signal[np.array(list(noise_idx))]*40, marker='x', color='red', label='Other spikes (artifacts)') # black

			except TypeError:
				print('noise spikes vector is empty, continue..')
			try:
				t_th = t_start + np.arange(0,len(threshold[1]))/self.fs
				th_df = pd.DataFrame(t_th, columns=['t'])
				th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
				axes.plot(th_df.index, np.array(threshold[1]), '-', linewidth=0.5, label='neg th', color='black') 
				axes_th.plot(th_df.index, np.array(threshold[1]), '-', linewidth=0.5, label='pos th', color='black')
			except TypeError:
				print('Cannot plot threshold')
			try:
				t_th = t_start + np.arange(0,len(threshold[0]))/self.fs
				th_df = pd.DataFrame(t_th, columns=['t'])
				th_df.index = pd.DatetimeIndex(th_df.t * 1e9)
				axes.plot(th_df.index, np.array(threshold[0]), '-', linewidth=0.5, label='pos th', color='black')
				axes_th.plot(th_df.index, np.array(threshold[0]), '-', linewidth=0.5, label='pos th', color='black')
			except TypeError:
				print('Cannot plot threshold') 
			axes.set_title('%s' %signal.name)
		else:
			pass

		# Format axes
		for ax in [axes, axes_th]:
			ax.set_xlabel('Time')
			ax.set_ylabel('Voltage [uV]')
			ax.set_ylim(ylim)
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			ax.xaxis.set_major_formatter(md.DateFormatter(dtformat))
			if plot_legend==True:
				ax.legend(prop={"size":10})
			
		# Get current time for saving (avoid overwriting)
		now = datetime.datetime.now()
		current_time = now.strftime("%d%m%Y_%H%M%S")
		if save_figure=='png':
			fig_th.savefig('%s/figures/threshold_evolution-%s-%s.png' %(self.path, ch,current_time), facecolor='w')
			fig.savefig('%s/figures/identified_peaks-%s-%s.png' %(self.path, ch,current_time), facecolor='w')
		elif save_figure=='svg':
			fig.savefig('%s/figures/identified_peaks-%s-%s.svg' %(self.path, ch,current_time), facecolor='w')
			fig_th.savefig('%s/figures/threshold_evolution-%s-%s.svg' %(self.path, ch,current_time), facecolor='w')

	def plot_cluster(self, data,ch, cluster, roll_win=True, window=1, units='s', dtformat='%M:%S.%f', show_plot=False, save='png', time_marks=[]): 
		"""
		Parameters:
		-----------
		cluster: vector of length equal to the number of peaks identified, with numbers corresponding to the cluster type.
		"""
		print('Plot cluster - Saving: %s' %save)
		# Get current time for saving (avoid overwriting)
		now = datetime.datetime.now()
		current_time = now.strftime("%d%m%Y_%H%M%S")

		# Plot the result
	
		# Unify colors of three plots:
		NUM_COLORS = len(np.unique(cluster))
		colors=[]
		cm = pylab.get_cmap('tab20c')  #https://matplotlib.org/stable/tutorials/colors/colormaps.html
		for i in range(NUM_COLORS):
			colors.append(cm(1.*i/NUM_COLORS)) # color will now be an RGBA tuple
		#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Default color list in python
		self.colors = colors
		#colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

		mycolors = []
		for i,c in enumerate(cluster):
			mycolors.append(self.colors[c])

		fig, ax = plt.subplots(1, 2, figsize=(10, 5))
		ax0 = fig.add_subplot(1, 2, 1, projection='3d')
		fig.suptitle('Cluster method: %s' %self.clustering_method)

		self.num_clusters = len(np.unique(cluster))
		
		# Identify cluster number in plot
		cluster_type = range(len(np.unique(cluster)))
		cluster_type_name = ["Cluster "+"{:02d}".format(x) for x in cluster_type]

		ax0.scatter(data[:, 0], data[:, 1], data[:, 2], c=mycolors, label=cluster_type_name)
		ax0.set_xlabel('1st component', fontsize=14)
		ax0.set_ylabel('2nd component', fontsize=14)
		ax0.set_zlabel('3rd component', fontsize=14)
		ax0.set_title('Clustering  (%s)' %(ch), fontsize=16)

		time = np.linspace(0, self.waveforms.shape[1] / self.fs, self.waveforms.shape[1]) * 1000

		for a, i in enumerate(np.unique(cluster)):
			cluster_mean = self.waveforms[cluster == i, :].mean(axis=0)
			cluster_std = self.waveforms[cluster == i, :].std(axis=0)

			ax[1].plot(time, cluster_mean, label='Cluster {}'.format(i), color=self.colors[i])
			ax[1].fill_between(time,
							   cluster_mean - cluster_std,
							   cluster_mean + cluster_std,
							   alpha=0.15, color=self.colors[i])
		
		ax[1].set_title('Average waveforms of identified clusters (%s)' %ch, fontsize=16)
		ax[1].set_xlim([0, time[-1]])
		ax[1].set_ylim([self.waveforms.min(axis=0).min(), self.waveforms.max(axis=0).max()])
		ax[1].set_xlabel('Time [ms]', fontsize=14)
		ax[1].set_ylabel('Voltage [uV]', fontsize=14)
		ax[1].legend()

		# Format axes
		for i in range(len(ax)):
			# Hide the right and top spines
			ax[i].spines['right'].set_visible(False)
			ax[i].spines['top'].set_visible(False)
			# Only show ticks on the left and bottom spines
			ax[i].yaxis.set_ticks_position('left')
			ax[i].xaxis.set_ticks_position('bottom')
		# Only for Scatter plot: 
		ax[0].spines['left'].set_visible(False)
		ax[0].spines['bottom'].set_visible(False)
		ax[0].get_xaxis().set_ticks([])
		ax[0].get_yaxis().set_ticks([])

		# Save
		if save=='png':
			fig.savefig('%s/figures/%s_clustering-%s.png' %(self.path, ch, current_time))
		elif save=='svg':
			fig.savefig('%s/figures/%s_clustering-%s.svg' %(self.path, ch, current_time))
			
		if show_plot==True:
			plt.show()
		else:
			print('Plot will not show')
			plt.close()

		if roll_win:
			#try:
				fig2 = plt.figure(figsize=(10, 10))
				fig3 = plt.figure(figsize=(10, 10))
				fig4 = plt.figure(figsize=(10, 10))
				#ax_r = plt.gca()
				#fig2, ax_r = plt.subplots(self.num_clusters, 1, figsize=(15, 8), sharex=True)
				fig2.suptitle('Num spikes in window %s ' %ch, fontsize=16, family='serif') 
				#fig3, ax_3 = plt.subplots(self.num_clusters, 1, figsize=(15, 8), sharex=True)
				fig3.suptitle('Amplitude of spikes %s ' %ch, fontsize=16, family='serif')  
				fig4.suptitle('Spike rate %s ' %ch, fontsize=16, family='serif')  
				for c in self.unique_clusters:
					""" Comented on 13/01/22 because it's duplicated in cluster_rolling_window above
					self.cluster_dataframe['rc_%s' %c] = self.cluster_dataframe['cluster_%s' %c].rolling(window='%s%s'%(window, units),
					  																			min_periods=int(self.fs * window)).sum()
					"""
					ax_r = fig2.add_subplot(self.num_clusters, 1,c+1)
					ax_r.plot(self.cluster_dataframe['rc_%s' %c], marker='o', markersize=0.3, label='%s'%c, color=colors[c])
					ax_r.set_xlabel('Time', fontsize=14)
					ax_r.set_ylabel('Spikes in %s sec' %window, fontsize=14)

					ax_sr = fig4.add_subplot(self.num_clusters, 1,c+1)
					ax_sr.plot(self.cluster_dataframe['src_%s' %c], marker='o', markersize=0.3, label='%s'%c, color=colors[c])
					ax_sr.set_xlabel('Time', fontsize=14)
					ax_sr.set_ylabel('Spike rate (Hz)', fontsize=14)

					ax_3 = fig3.add_subplot(self.num_clusters, 1,c+1)
					ax_3.plot(self.cluster_dataframe['metric_amplitude_c%s' %c] , marker='o', markersize=1, label='%s'%c, color=colors[c])
					#ax_r[c,1].set_title('Mean amplitude c_%s' %c, fontsize=14)
					ax_3.set_xlabel('Time', fontsize=14)
					ax_3.set_ylabel('Voltage (uV)', fontsize=14)
					
					# Format axes
					for ax in [ax_r, ax_sr, ax_3]:
						ax.xaxis.set_major_formatter(md.DateFormatter(dtformat))
						ax.spines['right'].set_visible(False)
						ax.spines['top'].set_visible(False)
						ax.yaxis.set_ticks_position('left')
						ax.xaxis.set_ticks_position('bottom')
						ax.legend()

				if save:
					if save=='png':
						fig2.savefig('%s/figures/%s_clustering_evolution-%s.png' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
						fig3.savefig('%s/figures/%s_clustering_amplitude_evolution-%s.png' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
						fig4.savefig('%s/figures/%s_clustering_spikerate_evolution-%s.png' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
					elif save=='svg':
						fig2.savefig('%s/figures/%s_clustering_evolution-%s.svg' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
						fig3.savefig('%s/figures/%s_clustering_amplitude_evolution-%s.svg' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
						fig4.savefig('%s/figures/%s_clustering_spikerate_evolution-%s.svg' %(self.path, ch, current_time))
						if show_plot==True:
							plt.show()
						else:
							print('Plot will not show')
							plt.close()
			#except ValueError:
			#	print('Cannot plot cluster evolution')

		

class MyWavelet:
	# Class created only to set the parameters for the wavelet analysis, no computations
	def __init__(self, wvl_type):
		self.wvl_type = wvl_type
		
	def set_dwt_kwargs(self, wavelet, start_level, end_level, level, thres_type,k, DBplot=False):
		self.kwargs = {'wavelet': wavelet, 
					  'start_level':start_level,
					  'end_level': end_level,
					  'level': level,
					  'thres_type': thres_type,
					  'k': k,
					  'DBplot': DBplot}

	def set_cwt_kwargs(self, wavelet, twave, tinterval, scales):
		self.kwargs = {'wavelet': wavelet, 
					  'twave':twave,
					  'tinterval': tinterval,
					  'scales':scales}

	def set_thres_param(self, thres_type, DBplot=False):
		self.thres_type = thres_type
		self.DBplot = DBplot


class Metabolic:
	def __init__(self, data):
		self.data = data

	@classmethod
	
	def open_metabolic_file(cls, path, port, fs=None, start=0, end=1000, nsamples=None, baseline_gluc=0, baseline_ins=0, baseline_glucgn=0, baseline_hr=0,verbose=1):

		# Load dataframes
		'''
		filepath = askopenfile(initialdir=path, title="Select metabolic file",
									filetypes=[("_bsamples", ".csv")])
		'''
		filepath = path
		#data = load_bsamples(filepath, fs = fs, start=start, nsamples=nsamples, verbose=verbose)
		data = load_bsamples_start_end(filepath, fs = fs, start=start, end=end, nsamples=nsamples, verbose=verbose)
		#print(data)
		data.index = pd.DatetimeIndex(data.seconds * 1e9)
		data.index.name = 'time'

		data.insulin[data['insulin'] == 0] = np.nan
		data.glucagon[data['glucagon'] == 0] = np.nan

		data['val_gluc'] = data.glucose#.interpolate(method='linear', limit=20, axis=0)
		#print(data['change_label'])
		#sys.exit()
		#baseline_gluc = data.glucose.between_time(baseline_onset, baseline_end).mean()
		baseline_gluc = baseline_gluc
		data['perc_base_gluc'] = 100 * ((data['val_gluc']-baseline_gluc)/baseline_gluc)
		data['pct_gluc'] = data['val_gluc'].pct_change()*100  # (row - previous)/previous) (not multiplied by 100)
		data['change_gluc_base'] = (data['val_gluc']- baseline_gluc)
		data['change_gluc'] = np.gradient(data['val_gluc']) #https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
		# a commonly used introductory technique to approximate the derivative of a sampled signal is to use a (symmetric) finite difference 
		#algorithm with an order of accuracy of 2. The equation of this would be:

		'''
		# Uncomment when you have insulin and glucagon data
		data['val_ins'] = data.insulin.interpolate(limit_direction='forward', axis=0)
		#baseline_ins = data.insulin.between_time(baseline_onset, baseline_end).mean()
		baseline_ins = baseline_ins
		data['perc_base_ins'] = 100 * ((data['val_ins']-baseline_ins)/baseline_ins)
		data['pct_ins'] = data['val_ins'].pct_change()
		data['change_ins'] = (data['val_ins']-baseline_ins)

		data['val_glucgn'] = data.glucagon.interpolate(limit_direction='forward', axis=0)
		#baseline_glucgn = data.glucagon.between_time(baseline_onset, baseline_end).mean()
		baseline_glucgn = baseline_glucgn
		data['perc_base_glucgn'] = 100 * ((data['val_glucgn']-baseline_glucgn)/baseline_glucgn)
		data['pct_glucgn'] = data['val_glucgn'].pct_change()
		data['change_glucgn'] = (data['val_glucgn']-baseline_glucgn)

		
		# Load HR signal saved from script
		hr = pd.read_pickle('%s/heart_rate_%s.pkl' %(path, port))

		data['val_hr'] = data.HR.interpolate(limit_direction='forward', axis=0)
		baseline_hr = baseline_hr
		data['perc_base_hr'] = 100 * ((data['val_hr']-baseline_hr)/baseline_hr)
		data['pct_hr'] = data['val_hr'].pct_change()
		data['change_hr'] = (data['val_hr']-baseline_hr)
		'''
		# Load electrode map
		Tk().withdraw()  # keep the root window from appearing

		print("Metabolic data loaded succesfully.")

		return cls(data)



	def compute_correlations(self, cluster_df, metabolic_df, t_int=0.5):
		# t_int time interval around metabolic time to compute mean value of cluster
		df_corr = pd.DataFrame(columns=['gluc', 'ins', 'glucgn','hr'])
		for c in self.unique_clusters:
			# First get mean value of period of t_int seconds around metabolic times
			self.data['mean_rc_%s'%c] = np.zeros(len(self.data.index))
			for i,t in enumerate(self.data.index):
				# Get the closest index to metabolic times
				#idx_time = cluster_df.index[cluster_df.index.get_loc(t, method='nearest')]  
				idx = cluster_df.index.get_loc(t, method='nearest')
				# Get the value corresponding to that index (not used, just for reference)
				# val.append(cluster_df.iloc[cluster_df.index.get_loc(t, method='nearest')])
				# Compute the mean value 'record.fs*t_int' samples around the time
				if int(idx-self.fs*t_int)<0:
					pass
				else:
					self.data['mean_rc_%s'%c].iloc[i] = (cluster_df.iloc[:,c].iloc[int(idx-self.fs*t_int):int(idx+self.fs*t_int)]).mean()
			# Compute glucose correlation for each cluster
			corr_gluc = metabolic_df.iloc[:,0].corr(self.data['mean_rc_%s'%c])
			# Compute insulin correlation for each cluster
			corr_ins = metabolic_df.iloc[:,1].corr(self.data['mean_rc_%s'%c])
			# Compute glucagon correlation for each cluster
			corr_glucgn = metabolic_df.iloc[:,2].corr(self.data['mean_rc_%s'%c])
			# Compute HR correlation for each cluster
			corr_hr = metabolic_df.iloc[:,3].corr(self.data['mean_rc_%s'%c])
			df_corr.loc['c%s'%c] = [corr_gluc, corr_ins, corr_glucgn, corr_hr]
		return df_corr


	def plot_metabolic_cluster(self, cluster_df, metabolic_df, dtformat='%M:%S.%f', figsize=(10, 15)):
		colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Default color list in python
		fig, ax = plt.subplots(self.num_clusters+4, 1, figsize=figsize,  sharex=True)
		#ax = ax.flatten()
		for c in self.unique_clusters:
			ax[c].plot(cluster_df.iloc[:,c], colors[c])
			#ax[c].set_title('Cluster %s' %c)
			ax[c].set_ylabel('Cluster %s \n Spike rate (spk/s)'%c, fontsize=12)
		ax[c+1].plot(metabolic_df.iloc[:,0], marker='o', color='red')
		#ax[c+1].set_title('Glucose')
		ax[c+1].set_ylabel('Glucose \n mg/dL', fontsize=12)
		ax[c+2].plot(metabolic_df.iloc[:,1], marker='o', color='green')
		#ax[c+2].set_title('Insulin')
		ax[c+2].set_ylabel('Insulin \n ng/mL', fontsize=12)
		ax[c+3].plot(metabolic_df.iloc[:,2], marker='o', color='purple')
		#ax[c+3].set_title('Glucagon')
		ax[c+3].set_ylabel('Glucagon \n pg/mL', fontsize=12)
		ax[c+4].plot(metabolic_df.iloc[:,3], marker='o')
		#ax[c+3].set_title('Glucagon')
		ax[c+4].set_ylabel('HR \n bpm', fontsize=12)
		
		for i in range(len(ax)):
			# Hide the right and top spines
			ax[i].spines['right'].set_visible(False)
			ax[i].spines['top'].set_visible(False)

			# Only show ticks on the left and bottom spines
			ax[i].yaxis.set_ticks_position('left')
			ax[i].xaxis.set_ticks_position('bottom')

			# Format axes
			ax[i].xaxis.set_major_formatter(md.DateFormatter(dtformat))



#--------------------------------------------------------
# Example

if __name__ == '__main__':
	# ----------------
	# Load data
	# ----------------
	# Path
	path = ('../datasets/feinstein/')
	file = 'IL1RKO_TNF--IL1B_4.27.2016_01__35-38min.mat'
	record = Recording.open_record(path, file, start=0, dur=6000000)
	record.set_gain(20)

	print(record.fs)
	filt_config = {
		'W': [100, ],
		'None': {},
		'butter': {
				'N': 6,				# The order of the filter
				'btype': 'high'   	# The type of filter.
		},	  
		'fir': {
				'n': 4,
		}  
	}

	spike_detection_config = {
		'spike_window': [int(0.0018 * record.fs), int(0.0018 * record.fs)],
		'C': 8,
		'find_peaks_args': {
		# Required minimal horizontal distance (>= 1) in samples between 
		# neighbouring peaks. Smaller peaks are removed first until the 
		# condition is fulfilled for all remaining peaks.
		'distance': 0.0018 * 2 * record.fs
		}
	}

	# General configuration
	apply_filter = 'None'	#'butter', 'fir', 'None'

	detect_method = "get_spikes_method" # "get_spikes_method", 'get_spikes_threshCrossing', 'so_cfar'
	thresh_type = "positive" # positive, negative, both_thresh

	# Create wavelet objects
	neural_wavelet = MyWavelet('dwt')
	neural_wavelet.set_dwt_kwargs(wavelet='db3',  start_level=1, 
									end_level=5, level=8)

	neural_wavelet.set_thres_param(thres_type="hard", DBplot=False) # If DWT with threshold

	cardiac_wavelet = MyWavelet('cwt')
	cardiac_wavelet. set_cwt_kwargs(wavelet='mexh', twave=0.000025)

	# -------------------------------------
	# Filter data 
	# -------------------------------------
	# Configure filter
	filt_config['butter']['Wn'] = fnorm(filt_config['W'], fs=record.fs).tolist() # The critical frequency or frequencies.
									# Same as doing (filt_config['W']/(record.fs/2)).tolist()
	
	record.filter(apply_filter, **filt_config[apply_filter])

	# ----------------------------------------
	# PLOT
	#-----------------------------------------
	""" To do
	- Check that filtering is working for few channels individually
	- Plot the raw channels and after filtering

	fig, axes = plt.subplots(11,3, sharex=True)
	for i,ax in enumerate(axes.flatten()):
		if np.isnan(record.map_array[i]):
			pass
		else:
			ax.plot(record.recording['ch_%s'%int(record.map_array[i])], lw=0.5)

	plt.show()
	sys.exit()

	"""

	# ---------------------------------------
	# Wavelet decomposition 
	# ---------------------------------------
	print("----------------------")
	print(" Neural decomposition")
	
	# Single channel
	neural_wvl = wavelet_decomp(record.recording, record.fs, 
								type=neural_wavelet.wvl_type,
								**neural_wavelet.kwargs)
	record.wavelet_results['wavelet_decomp'] = neural_wvl

	""" Adaptive noise removal: Thresholding detail coefficients"""
	print(record.recording)
	neural_wvl_denoised = DWT_with_denoising(record.recording.neural_b, record.fs,
											 thres_type=neural_wavelet.thres_type,
											 DBplot=neural_wavelet.DBplot,
											 **neural_wavelet.kwargs)
	record.wavelet_results['DWT_with_denoising'] = neural_wvl_denoised

	print("----------------------")
	print(" Cardiac decomposition")
	cardiac_wvl = wavelet_decomp(record.recording, record.fs, 
								type=other_wavelet.wvl_type,
								**other_wavelet.kwargs)

	record.recording = neural_wvl_denoised
	# ----------------------------------------------------
	# Extract spikes from the filtered neural signal
	# -----------------------------------------------------
	"""
	Check with cwt signal taken directly from Matlab Feinstein

	path = ('../datasets/feinstein/cwt_reference.mat') # feinstein.mat

	# Load dataframes
	data = sp.io.loadmat(path)
	neural_wvl = data['sig'].reshape(-1)
	print(neural_wvl)
	s = pd.DataFrame(neural_wvl)
	"""

	print("-------------------------------")
	print("Detecting neural events")

	spikes_idx, wave_form, numunique_peak_pos, num_noise_peaks = \
	 record.spike_detection(record.recording, detect_method, thresh_type, **spike_detection_config)
	
	spikes_vector_ampl = record.get_spike_amplitude(record.recording, spikes_idx)
	spikes_vector_loc = record.get_spike_location(record.recording, spikes_idx)

	plt.figure()
	plt.plot(record.recording, '-', linewidth=0.5, label='Signal')
	plt.scatter(np.array(list(spikes_idx)), spikes_vector_ampl[np.array(list(spikes_idx))], marker='o', label='spikes')
	plt.plot(np.array(record.threshold[1]), '-', linewidth=0.5, label='neg th')
	plt.plot(np.array(record.threshold[0]), '-', linewidth=0.5, label='pos th')

	plt.legend(loc='upper right')

	plt.show()


	######################################
	# SPC
	#######################################
	randomseed = 0
	rng = np.random.RandomState(randomseed)

	cl1 = rng.multivariate_normal([8, 8], [[4,0],[0,3]], size=800)
	cl2 = rng.multivariate_normal([0,0], [[3,0],[0,2]], size=2000)
	cl3 = rng.multivariate_normal([5,5], [[0.5,0.2],[0.2,0.6]], size=300)
	cl4 = rng.multivariate_normal([-3,-2.5], [[0.5,0],[0,0.6]], size=500)
	gt = [cl1,cl2,cl3,cl4]

	#plot clusters
	plt.figure()
	for cl in gt:
		plt.plot(*cl.T,marker='.',linestyle='',markersize=3)
	plt.title('Ground Truth')
	plt.grid()
	plt.show()


	#run the algorithm. The fit method applied the cluster selection described in Waveclus 3. The method fit_WC1 is the alternative using the original Waveclus 1 temperature selection.
	data = np.concatenate(gt)
	clustering = SPC(mintemp=0,maxtemp=0.4,randomseed=randomseed)
	labels, metadata = fit_WC3(data,clustering,min_clus=150,return_metadata=True)

	#It is posible to show a temperature map using the optional output metadata
	#plot_temperature_plot(metadata)
	#plt.show()

	#To show the assigned labels:
	plt.figure()
	for c in np.unique(labels):
		if c==0:
			plt.plot(*data[labels==c,:].T,marker='.',color='k',linestyle='',markersize=3)
		else:
			plt.plot(*data[labels==c,:].T,marker='.',linestyle='',markersize=3)
	plt.grid()
	plt.title('Results')
	plt.show()