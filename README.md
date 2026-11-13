
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

## Run the classification benchmark

1. Check out the `classification` branch and open a terminal in the repository
   root.
2. On macOS, install the OpenMP runtime for XGBoost with `brew install libomp`.
3. Create a Python 3.11 environment and install the pinned dependencies.

   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

4. Confirm the Bosch recordings are under `data/` with their original directory
   layout. If needed, download them from the dataset source above.
5. Open `classification.ipynb` from the repository root in VS Code or Jupyter.
   Select the `.venv` kernel and run all cells. The final tables show both
   protocols, including per-class training and holdout window counts.

The notebook extracts 100 time- and frequency-domain features from X, Y, Z and
their combined magnitude. Each recording is divided into non-overlapping
4,000-sample windows (two seconds at 2 kHz). The notebook uses every complete
window and drops the shorter tail.

| Protocol | Training | Evaluation |
| --- | --- | --- |
| Anomaly-style window split | Healthy windows equal to 80% of all windows | Remaining healthy and all faulty windows, divided into selection and final test |
| Recording split | Approximately 80% of healthy recordings | Remaining healthy recordings, with zero overlap in source files |

The window split can put windows from the same recording in both training and
evaluation. Sorted paths and seed 42 reproduce the split. The recording split
measures performance on unseen healthy recordings. Its test population differs
from the window split, which also includes faulty windows.

Random Forest retains the original unweighted settings with 300 trees, up to
1,024 bootstrap samples per tree, and a 0.7 feature fraction at each split.
XGBoost uses 300 histogram-based boosting rounds and inverse-frequency sample
weights computed separately for each target from its training labels.
Both models use the same features and rows within each protocol and target.
The notebook uses fixed seeds and trains on the CPU.

Accuracy counts every holdout window. Macro F1 averages across classes with
holdout support, matching the original Random Forest report. A class missing
from training remains in the holdout and receives zero F1. XGBoost needs at
least two training classes, so the notebook marks that fit unavailable when
only one is present. Fit time excludes feature extraction, weight calculation,
and prediction. The two anomaly holdouts share one fit per model and target.

The previously recorded Random Forest machine baseline was **91.0% accuracy**
and **63.4% macro F1** on the anomaly-style final test.

Run the checks with `python -m unittest discover -s tests -v`.

## Measured Bosch results

One run on an Apple M2 processed 1,702 recordings and 43,847 complete windows.
The anomaly selection and final test shared 761 and 818 recordings with their
training set, respectively. The recording test included 8,166 windows from
327 healthy recordings, with zero recordings shared with training.

| Protocol | Target | Model | Holdout | Accuracy | Macro F1 | Fit (s) |
| --- | --- | --- | --- | --- | --- | ---: |
| anomaly | machine | Random Forest | selection | 89.58% | 90.78% | 5.107 |
| anomaly | machine | Random Forest | final test | 90.99% | 63.43% | 5.107 |
| anomaly | machine | XGBoost | selection | 94.84% | 95.25% | 9.428 |
| anomaly | machine | XGBoost | final test | 96.24% | 81.96% | 9.428 |
| anomaly | operation | Random Forest | selection | 71.53% | 65.55% | 4.904 |
| anomaly | operation | Random Forest | final test | 68.51% | 64.05% | 4.904 |
| anomaly | operation | XGBoost | selection | 84.53% | 81.87% | 43.560 |
| anomaly | operation | XGBoost | final test | 84.95% | 81.66% | 43.560 |
| recording | machine | Random Forest | test | 95.53% | 95.48% | 4.278 |
| recording | machine | XGBoost | test | 98.27% | 98.23% | 7.818 |
| recording | operation | Random Forest | test | 76.32% | 73.69% | 4.839 |
| recording | operation | XGBoost | test | 92.02% | 91.21% | 41.042 |
