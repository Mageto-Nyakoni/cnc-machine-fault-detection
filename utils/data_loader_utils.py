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
        


def time_features(samples):
    """Compute main's 13 time-domain features for one signal."""
    samples = np.asarray(samples, dtype=float)
    centered = samples - samples.mean()
    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))
    abs_mean = np.mean(np.abs(samples))
    varying = np.ptp(samples) > 0
    return {
        "mean": samples.mean(),
        "std": samples.std(ddof=1) if len(samples) > 1 else 0.0,
        "min": samples.min(),
        "max": samples.max(),
        "ptp": np.ptp(samples),
        "rms": rms,
        "abs_mean": abs_mean,
        "skew": stats.skew(samples) if varying else 0.0,
        "kurtosis": stats.kurtosis(samples) if varying else 0.0,
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "shape_factor": rms / abs_mean if abs_mean > 0 else 0.0,
        "impulse_factor": peak / abs_mean if abs_mean > 0 else 0.0,
        "zero_crossing_rate": (
            np.mean(np.diff(np.sign(centered)) != 0) if len(samples) > 1 else 0.0
        ),
    }


def freq_features(samples, fs=2000, nperseg=1024):
    """Compute main's 12 Welch-spectrum features for one signal."""
    samples = np.asarray(samples, dtype=float)
    samples = samples - samples.mean()
    frequencies, psd = signal.welch(
        samples, fs=fs, nperseg=min(nperseg, len(samples)), window="hann"
    )
    psd_sum = psd.sum()
    probabilities = psd / psd_sum if psd_sum > 0 else np.ones_like(psd) / len(psd)
    centroid = np.sum(frequencies * probabilities)
    spread = np.sqrt(np.sum((frequencies - centroid) ** 2 * probabilities))
    cumulative = np.cumsum(probabilities)
    frequency_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else fs
    features = {
        "dominant_freq": frequencies[np.argmax(psd)],
        "peak_psd": psd.max(),
        "total_power": psd_sum * frequency_step,
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_skew": np.sum((frequencies - centroid) ** 3 * probabilities)
        / (spread ** 3 + 1e-12),
        "spectral_kurtosis": np.sum((frequencies - centroid) ** 4 * probabilities)
        / (spread ** 4 + 1e-12),
        "spectral_entropy": -np.sum(probabilities * np.log2(probabilities + 1e-12)),
        "spectral_rolloff_85": frequencies[np.searchsorted(cumulative, 0.85)],
    }
    bands = {"low": (0, fs / 20), "mid": (fs / 20, fs / 8), "high": (fs / 8, fs / 2)}
    for name, (lower, upper) in bands.items():
        mask = (frequencies >= lower) & (frequencies < upper)
        features[f"band_power_{name}"] = (
            np.trapezoid(psd[mask], frequencies[mask]) if mask.any() else 0.0
        )
    return features


def extract_window_features(window, fs=2000):
    """Return 100 time/frequency features for X, Y, Z and their magnitude."""
    values = np.asarray(window, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("Expected a nonempty array with X, Y, Z columns.")

    if not np.isfinite(values).all():
        raise ValueError("Window contains missing or infinite values.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be finite and positive.")

    features = {}
    axes = {axis: values[:, column] for column, axis in enumerate(("x", "y", "z"))}
    axes["mag"] = np.sqrt((values ** 2).sum(axis=1))
    for axis, samples in axes.items():
        for name, value in time_features(samples).items():
            features[f"{axis}_{name}"] = value
        for name, value in freq_features(samples, fs).items():
            features[f"{axis}_{name}"] = value
    return features


if __name__ == "__main__":
    data_path = Path("./EECE5644_Project/data").absolute()
    file_list = find_all_h5s_in_dir(data_path)
    print(file_list[0])
    file_df = format_file(file_list[0])
    print(file_df)
    
