# Nonlinear Machine Learning Models

A Python project exploring **nonlinear classification** and **polynomial regression** using NumPy, scikit-learn, and Matplotlib.

## Features

- Compare linear classification with L2-norm and polynomial feature transformations on a rings dataset.
- Visualize classification results and decision boundaries.
- Fit polynomial regression models of different degrees to two datasets.
- Compare training and test performance using classification error and mean squared error (MSE).

## Project structure

```text
nonlinear-machine-learning-models/
├── src/
│   ├── nonlinear_classification.py
│   └── polynomial_regression.py
├── data/
│   ├── rings.npz
│   ├── poly_s.npz
│   └── poly_d.npz
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

Install dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The original scripts load data using relative filenames such as `rings.npz`, so **run them from the `data/` directory**:

```bash
cd data
python3 ../src/nonlinear_classification.py
python3 ../src/polynomial_regression.py
```

The scripts print model evaluation results and display plots.

## Technologies

Python · NumPy · scikit-learn · Matplotlib
