
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
