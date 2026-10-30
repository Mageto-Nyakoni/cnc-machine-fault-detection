#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""





"""

""" 
# Copyright (c) 2021 Bosch Rexroth AG
# SPDX-License-Identifier: BSD-3-Clause

Author: Mohamed-Ali Tnani (Mohamed-ali.tnani@boschrexroth.de)
Date: November 24, 2021

This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright holder or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from typing import Literal
from pathlib import Path

def find_all_h5s_in_dir(s_dir: os.PathLike):
    """
    list all .h5 files in a directory with their complete path
    """

    fileslist = []
    for root, dirs, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(root+"/"+file)
    return fileslist


def load_tool_research_data(data_path: os.PathLike, label: Literal["good","bad"], add_additional_label: bool =True, verbose: bool =True):
    """
    load data (good and bad) from the research data storages
    
    Keyword Arguments:
            data_path {str} -- [path to the directory] 
            label {str} -- ["good" or "bad"]
            add_additional_label {bool} -- [if true the labels will be in the form of "Mxx_Aug20xx_OPxx_sampleNr_label" otherwise "label"] (default: True)
            verbose {bool}

        Returns:
            datalist --  [list of the the X samples]
            label --  [list of the the y labels ]
    """
    datalist = []
    data_label = []

    # list all .h5 files
    list_paths = find_all_h5s_in_dir(data_path)
    list_paths.sort()
    if not list_paths and verbose:
        print(f"skipping {data_path} empty directory...")

    # read and append the samples with the corresponding labels
    if verbose:
        print(f"loading files from {data_path}... ")
    for element in list_paths:
        # check if additional label needed ("Mxx_Aug20xx_Tool,nrX") 
        if add_additional_label:
            add_label = element.split('/')[-1]
            additional_label = add_label[:-3] + "_" + label
        else:
            additional_label = label
        # extract data X and y 
        with h5py.File(os.path.join(data_path, element), 'r') as f:
            vibration_data = f['vibration_data'][:]
        datalist.append(vibration_data)
        data_label.append(additional_label)

    return datalist, data_label


def datafile_read(file: os.PathLike, plotting=True):
    """loads and plots the data from the datafile

    Keyword Arguments:
        file {str} -- [path of the file] 

    Returns:
        ndarray --  [raw data original]
    """
    with h5py.File(file, 'r') as f:
        vibration_data = f['vibration_data'][:]
    # interpolation for x axis plot
    freq = 2000
    samples_s = len(vibration_data[:, 0]) / freq
    samples = np.linspace(0, samples_s, len(vibration_data[:, 0]))

    # plotting
    if plotting:
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 0], 'b')
        plt.ylabel('X-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.show()
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 1], 'b')
        plt.ylabel('Y-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.show()
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 2], 'b')
        plt.ylabel('Z-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.show()
    return vibration_data


def format_file(file: os.PathLike):
    """Converts h5 file data into a pandas df, adding additional class and metadata columns
    df = [X, Y, Z, timestamp, label, machine, process, i, window]
    
    Keyword Arguments:
        file {str} -- [path of the file] 

    Returns:
        pd DataFrame --  [formatted data table]
    """
    with h5py.File(file, 'r') as f:
        dataset = f['vibration_data'][:]
        df = pd.DataFrame(dataset, columns=["X","Y","Z"])
        # Define additional data columns
        df["timestamp"] = np.arange(len(df)) * 0.5
        file_name = str(file).split("/")[-1]
        label = str(file).split("/")[-2]
        df["label"] = label == "good"
        df["machine"] = int(file_name.split("_")[0].replace("M",""))
        df["process"] = int(file_name.split("_")[3].replace("OP",""))
        df["i"] = int(file_name.split("_")[4].replace(".h5",""))
        df["window"] = df["timestamp"] // 2000 # Windows with 4000 records, ~2s
        # Downcast datatypes to reduce memory usage
        df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="float")
        df["machine"] = pd.to_numeric(df["machine"], downcast="integer")
        df["process"] = pd.to_numeric(df["process"], downcast="integer")
        df["i"] = pd.to_numeric(df["i"], downcast="integer")
        df["window"] = pd.to_numeric(df["window"], downcast="integer")
        
    return df

def time_features(x):
    x = np.asarray(x, dtype=float)
    x_centered = x - x.mean()
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    return {
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "min": x.min(),
        "max": x.max(),
        "ptp": np.ptp(x),
        "rms": rms,
        "abs_mean": np.mean(np.abs(x)),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "crest_factor": peak / rms if rms > 0 else 0,
        "shape_factor": rms / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
        "impulse_factor": peak / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
        "zero_crossing_rate": np.mean(np.diff(np.sign(x_centered)) != 0),
    }
     
def freq_features(x, fs=2000, nperseg=1024):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()

    f, psd = signal.welch(x, fs=fs, nperseg=nperseg, window="hann")
    psd_sum = psd.sum()
    p = psd / psd_sum if psd_sum > 0 else np.ones_like(psd) / len(psd)

    centroid = np.sum(f * p)
    spread = np.sqrt(np.sum(((f - centroid) ** 2) * p))
    cumulative = np.cumsum(p)

    feats = {
        "dominant_freq": f[np.argmax(psd)],
        "peak_psd": psd.max(),
        "total_power": psd_sum * (f[1] - f[0]),
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_skew": np.sum(((f - centroid) ** 3) * p) / (spread**3 + 1e-12),
        "spectral_kurtosis": np.sum(((f - centroid) ** 4) * p) / (spread**4 + 1e-12),
        "spectral_entropy": -np.sum(p * np.log2(p + 1e-12)),
        "spectral_rolloff_85": f[np.searchsorted(cumulative, 0.85)],
    }

    bands = {"low": (0, fs/20), "mid": (fs/20, fs/8), "high": (fs/8, fs/2)}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        feats[f"band_power_{name}"] = np.trapezoid(psd[mask], f[mask]) if mask.any() else 0.0

    return feats
        
def extract_window_features(window, fs=2000):
    """ Generates a vector with the sample set variables for X, Y and Z given a 4,000 record window of vibration data.

    Args:
        window (np.array): 4,000 record window array of vibration data.
        fs (int, optional): Sample rate of the data in Hz. Defaults to 2000.

    Returns:
        feats (pd.DataFrame): Dataframe including all the feature set variables for X, Y and Z.
    """
    feats = {}
    axes = {"x": window[:, 0], "y": window[:, 1], "z": window[:, 2]}

    for name, sig in axes.items():
        for k, v in time_features(sig).items():
            feats[f"{name}_{k}"] = v
        for k, v in freq_features(sig, fs).items():
            feats[f"{name}_{k}"] = v

    # Combined magnitude of the 3 axes (orientation-independent)
    mag = np.sqrt((window ** 2).sum(axis=1))
    for k, v in time_features(mag).items():
        feats[f"mag_{k}"] = v
    for k, v in freq_features(mag, fs).items():
        feats[f"mag_{k}"] = v

    return feats


if __name__ == "__main__":
    data_path = Path("./EECE5644_Project/data").absolute()
    file_list = find_all_h5s_in_dir(data_path)
    print(file_list[0])
    file_df = format_file(file_list[0])
    print(file_df)
    