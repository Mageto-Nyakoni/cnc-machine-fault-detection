
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

The notebook extracts 100 time- and frequency-domain features from X, Y, Z and
their combined magnitude. Every recording is divided into non-overlapping
4,000-sample windows (two seconds at 2 kHz). The notebook uses every complete
window and drops the shorter tail, as the anomaly detection branch does.

The machine and operation Random Forests start with the anomaly branch's
highest-PR-AUC Isolation Forest settings: 300 trees, up to 1,024 training
windows per tree, a feature fraction of 0.7, and seed 42. The classifiers use
bootstrap sampling and apply the feature fraction at each split. The notebook
reports accuracy, precision, recall, and F1 for each target.

The default `SPLIT_MODE = "anomaly"` follows the anomaly notebook's window-level
split. Training draws healthy windows equal to 80% of all windows. The remaining
healthy windows and all faulty windows are divided between selection and final
test. The input paths are sorted for reproducibility, so the sampled window IDs
can differ from the anomaly notebook's filesystem-order run. The same source
recording can contribute to both training and evaluation.

Set `SPLIT_MODE = "recording"` to train on healthy recordings and hold out 20% of
those recordings as whole files. Only vibration features enter either model.
The notebook shows HDF5 read and formatting time, total feature-table preparation
time, and each model's fit time. Notebook outputs are cleared in Git.

Run feature checks with `python -m unittest discover -s tests -v`.
