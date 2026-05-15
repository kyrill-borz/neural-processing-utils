import os
import sys
import json
import time
import datetime
#import pycwt
import numpy as np
from scipy.stats import ks_2samp, expon
import numpy as np
import scipy as sp
import pandas as pd
import polars as pl
#import seaborn as sns
import sklearn as sk
#import imageio
import itertools
#import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
import pylab
#import umap
import umap.umap_ as umap
import matplotlib.pyplot as plt
#import tkinter as tk
#from tkinter import *
# from tkinter import simpledialog
from sklearn import metrics
#import statistics

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
from scipy.signal import find_peaks, correlate
#from scipy import ndimage
from scipy.stats import zscore
# TKinter for selecting files
from tkinter import Tk	 # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfile

# Bokeh
from bokeh.io import show #,output_notebook, show
from bokeh.plotting import figure
from math import pi
from bokeh.layouts import gridplot

# Superparamagnetic clustering
#from spclustering import SPC #, plot_temperature_plot  https://github.com/ferchaure/SPC

# Add my module to python path
sys.path.append("../")

# Own libraries
from utils.load_data import load_setup, load_matfiles, load_data_multich, load_bsamples,load_data_dask, load_bsamples_start_end
from utils import *
from utils.utils import _json_safe
# from visualization.graphics.plots import *
# from processing.filter import FIR_smooth
# from visualization.SC_topo import *
from matplotlib.ticker import FuncFormatter
from utils.autofilter import adaptive_filter
from dataclasses import dataclass


@dataclass
class RecordingData:
	def __init__(self, data: np.ndarray, fs: float, channels: list[str], ch_loc: np.ndarray, name: str, information: dict):
        self.data: pl.DataFrame= data
        self.name: str = name
		self.fs: float = fs
        self.channels: list[str] = channels
        self.ch_loc: np.ndarray = ch_loc
		self.information: dict = information
		self._is_lazy = isinstance(data, pl.LazyFrame)
        

class SignalPreprocessor:
    def __init__(self, recording: RecordingData):
        self.recording = recording
    
class Recording:
    def __init__(self, neural, fs, length, map_array, filename, information): # intan_ch, Z_magnitude, Z_phase):
		"""Constructor of neurogram object. 
		Takes parameters from classmethod open_record, and also initialises other parameters
		
		Parameters
		------------
		neural: LazyFrame or DataFrame (supports both lazy and eager execution)
		fs: frequency of recordings
		length: length of neural dataframe 
		map_array: 
		filename:
		intan_ch: [list of ints] Available channels from amplifier data in rhs file

		Return
		--------
		self: object that contains all the parameters

		"""
		#self.recording = neural.neural_b
		self.raw_recording = RecordingData(data=neural, fs=fs, channels=channels, ch_loc=ch_loc, name=filename, information=information)
		self.filename = filename
        self.map_array=map_array  # Intan channels corresponding to electrodes numbered 1 to 32: intan ch corresponding to electrode 1 = map_array[0]
		self.length = length
    
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
			neural_lazy,fs, basename_without_ext, information = load_data_multich(path, start=start, dur=dur, port=port,  #intan_ch, Z_magnitude, Z_phase 
												load_from_file=load_from_file,
												load_multiple_files=load_multiple_files,
												fileType=fileType, 
												downsample=downsample,
												day=day, # Added Feb 2024 to account for chonic data
											verbose=verbose,
											return_lazy=True)  # Keep as lazy for optimization
		
		# Note: We DO NOT collect here anymore - keep data lazy throughout pipeline
		# Data will be collected only when needed (e.g., for spike detection, display)
		print(type(neural_lazy).__name__)  # Print LazyFrame or DataFrame
		
		# For length, we need to get it from the lazy frame schema or peek at one collect
		if isinstance(neural_lazy, pl.LazyFrame):
			# To get length without full materialization, collect the entire thing 
			# Note: We have to do this once to know length, but then lazy operations reuse the cached data
			try:
				print("Getting length from LazyFrame...")
				neural_collected = neural_lazy.collect()
				length = len(neural_collected)
				# Convert back to lazy for memory efficiency
				neural_lazy = neural_collected.lazy()
				print(f"Length determined: {length} samples")
			except Exception as e:
				print(f"Could not determine length: {e}")
				length = 0
		else:
			length = len(neural_lazy)
		print(f"Estimated/Confirmed length: {length}")
		
		if map_path is not None:
			map_df = pd.read_csv(map_path)
			map_array = map_df.to_numpy()
		else:
			map_data = askopenfile(initialdir=path,title="Select electrode map csv file", filetypes=[("CSV files", "*.csv")])
			map_array = pd.read_csv(map_data.name)
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
		if length is not None:
			print('Recording length: %s(samples), %s(s): ' %(length, length/fs))
		else:
			print('Recording length: Will be determined on collection')
		# Pass lazy frame to constructor - no immediate materialization
		return cls(neural_lazy, fs, length, map_array, basename_without_ext, information)
	
	def export(self, folder_path):
		os.makedirs(folder_path, exist_ok=True)

		# Helper to collect if lazy before writing
		def collect_and_write(data, filepath):
			if isinstance(data, pl.LazyFrame):
				data.collect().write_parquet(filepath)
			elif isinstance(data, pl.DataFrame):
				data.write_parquet(filepath)

		if self.recording is not None:
			collect_and_write(
				self.recording,
				os.path.join(folder_path, "raw_signal.parquet")
			)
		if self.filtered is not None:
			collect_and_write(
				self.filtered,
				os.path.join(folder_path, "filtered.parquet")
			)
		if self.referenced is not None:
			collect_and_write(
				self.referenced,
				os.path.join(folder_path, "referenced.parquet")
			)
		spike_rows = []
		# for ch, spikes in self.spike_data["spike_times"].items():
		# 	for s in spikes:
		# 		spike_rows.append({"channel": ch, "time": float(s)})
		
		# if spike_rows:
		# 	pl.DataFrame(spike_rows).write_parquet(
		# 		os.path.join(folder_path, "spike_times.parquet")
        #     )

		metadata = {
			"fs": self.fs,
			"length": self.length,
			"map_array": self.map_array,
			"filename": self.filename,
			"information": [],

			"channels": self.channels,
			"filter_ch": self.filter_ch,
			"threshold": self.threshold,
		}
		safe_metadata = _json_safe(metadata)
		with open(os.path.join(folder_path, "metadata.json"), "w") as f:
			json.dump(safe_metadata, f, indent=2)

	@classmethod
	def load_from_folder(cls, folder_path):

		# --- Load metadata ---
		with open(os.path.join(folder_path, "metadata.json"), "r") as f:
			metadata = json.load(f)

		# --- Load main recording (REQUIRED) ---
		recording_path = os.path.join(folder_path, "raw_signal.parquet")

		if not os.path.exists(recording_path):
			raise ValueError("Missing raw_signal.parquet (required)")

		neural = pl.read_parquet(recording_path)

		# --- Construct object properly ---
		obj = cls(
			neural=neural,
			fs=metadata["fs"],
			length=metadata["length"],
			map_array=metadata["map_array"],
			filename=metadata["filename"],
			information=metadata["information"]
		)

		# --- Optional signals ---
		def load_optional(name):
			path = os.path.join(folder_path, name)
			return pl.read_parquet(path) if os.path.exists(path) else None

		obj.filtered = load_optional("filtered.parquet")
		obj.referenced = load_optional("referenced.parquet")

		# --- Spike data ---
		spike_path = os.path.join(folder_path, "spike_times.parquet")

		if os.path.exists(spike_path):

			df = pl.read_parquet(spike_path)

			spike_dict = {}

			for ch in df["channel"].unique():
				spike_dict[ch] = df.filter(
					pl.col("channel") == ch
				)["time"].to_numpy()

			obj.spike_data = {"spike_times": spike_dict}

		else:
			obj.spike_data = {"spike_times": {}}

		# --- Restore other attributes (important) ---
		obj.channels = metadata.get("channels", [])
		obj.filter_ch = metadata.get("filter_ch", [])
		obj.threshold = metadata.get("threshold", [])

		return obj
