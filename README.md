
# CNC Vibration Fault Detection

EECE 5644 - Machine Learning and Pattern Recognition  
Northeastern University, Fall 2026

## Team
- Eduardo Scopelli
- Mageto Nyakoni

## Project overview

We use CNC vibration data to identify machines and machining operations and
to detect vibration patterns that may indicate faults.

## Dataset

The Bosch CNC Machining Dataset contains 1,702 HDF5 recordings from three
machines running multiple operations. Each recording has X, Y, and Z
vibration measurements sampled at 2 kHz.
[Dataset source](https://github.com/boschresearch/CNC_Machining)

## Models

Random Forest and XGBoost classify machines and operations. The planned
anomaly detection models are One-Class SVM and Isolation Forest.

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
   Select the `.venv` kernel and run all cells. The last two tables show model
   scores and window counts by class.

The notebook divides each recording into non-overlapping two-second windows
of 4,000 samples and drops any shorter tail. For every complete window it
extracts 100 time- and frequency-domain features from X, Y, Z, and their
combined magnitude.

| Protocol | Training | Evaluation |
| --- | --- | --- |
| Anomaly-style window split | Healthy windows equal to 80% of all windows | Remaining healthy and all faulty windows, divided into selection and final test |
| Recording split | Approximately 80% of healthy recordings | Remaining healthy recordings, with zero overlap in source files |

The window split can put windows from the same recording in training and
evaluation. Sorting file paths and using seed 42 keeps the split reproducible.
Use the recording split to measure performance on unseen healthy recordings.
The window split also evaluates faulty windows.

Random Forest uses the original settings. It fits 300 trees with up to 1,024
bootstrap samples per tree, considers 70% of features at each split, and gives
all training windows equal weight. XGBoost uses 300 boosting rounds with
histogram-based splits. For each target, it gives more weight to windows
from rarer training classes. Both models use the same features and rows for
each target and split. They train on the CPU with fixed seeds.

Accuracy counts every holdout window. Macro F1 gives equal weight to each
class present in the holdout, as in the original Random Forest report. A class
that appears only in the holdout contributes zero F1. XGBoost requires at least
two training classes. With one class, the table marks the fit unavailable.

Fit time measures training only. For the window split, each model trains once
per target, then scores the selection and final test sets. Both rows show
the same fit time.

The previously recorded Random Forest machine baseline was **91.0% accuracy**
and **63.4% macro F1** on the anomaly-style final test.

Run the checks with `python -m unittest discover -s tests -v`.

## Measured Bosch results

On an Apple M2, the notebook processed 43,847 complete windows from 1,702
recordings. The window selection set shared 761 recordings with training.
The window final test shared 818. The recording test covered 8,166 windows
from 327 healthy recordings held out entirely from training.

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
