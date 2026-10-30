
# CNC Vibration Fault Detection

EECE 5644 - Machine Learning and Pattern Recognition  
Northeastern University, Fall 2026

## Team
- Eduardo Scopelli
- Mageto Nyakoni

## Project Overview
This project explores machine learning methods for
analyzing vibration data from CNC machines.

We aim to investigate:
1. Classification of machines and machining operations.
2. Anomaly detection for identifying abnormal vibration
   patterns associated with potential machine faults.

## Dataset
Source: Bosch CNC Machining Dataset
https://github.com/boschresearch/CNC_Machining

The dataset contains 1,702 HDF5 recordings from three
CNC machines across multiple machining operations,
with X, Y, and Z vibration measurements sampled at 2 kHz.

## Proposed Models
Classification:
- Random Forest
- XGBoost

Anomaly Detection:
- One-Class SVM
- Isolation Forest

## Getting Started
1. Clone this repository.
2. Create a Python virtual environment.
3. Install dependencies:
   pip install -r requirements.txt
4. Download the dataset from the source repository.
5. Place the data in the data/ directory.

## Project Status
Initial setup and exploratory data analysis.

## Run the classification baseline

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Open `classification.ipynb` with the repository root as the working directory.
3. Run all cells in order.

The notebook checks one window, then extracts 15 vibration features from each
complete 4,000-sample window in healthy recordings. It reports skipped incomplete
windows and class coverage before fitting separate Random Forest classifiers for
machine and operation identification. It prints accuracy, precision, recall, and
F1 scores for each target.

The split holds out 20% of recordings. Every window from a recording stays in
the same split, and only vibration features enter the models. These are window
classification results on unseen healthy recordings, not fault-detection results
or a test of performance on future timeframes. Notebook outputs are cleared in Git.

Run feature checks with `python -m unittest discover -s tests -v`.
