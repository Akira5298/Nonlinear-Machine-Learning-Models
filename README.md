# Nonlinear Machine Learning Models

A Python project exploring **nonlinear classification** and **polynomial regression** using NumPy, scikit-learn, and Matplotlib.

The project includes interactive terminal analyzers for selecting experiments, comparing training and test performance, visualizing results, and optionally exporting plots as PNG images.

## Features

- Compare logistic regression with L2-norm and degree-2 polynomial feature transformations on a rings dataset.
- Visualize decision boundaries for the full dataset or separate training and test sets.
- Fit polynomial regression models of degrees 0, 1, 2, and 5 to two datasets.
- Compare training and test performance using classification error and mean squared error (MSE).
- Select individual plots or run all experiments through interactive terminal menus, with optional uniquely named PNG exports.

## Project Structure

```text
nonlinear-machine-learning-models/
├── src/
│   ├── nonlinear_classification.py
│   └── polynomial_regression.py
├── data/
│   ├── rings.npz
│   ├── poly_s.npz
│   └── poly_d.npz
├── Media/                     # Selected figures shown below
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

Install dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Run either interactive analyzer from the repository root:

```bash
python3 src/nonlinear_classification.py
python3 src/polynomial_regression.py
```

Select a dataset or experiment and choose which plots to display. The analyzers print evaluation results and can optionally save figures to a `plots/` folder next to each script.

## Results

### Nonlinear Classification

The rings dataset illustrates why a linear decision boundary may be insufficient and how feature transformations enable nonlinear separation.

**Original Dataset**

![Original rings dataset](Media/rings_raw.png)

**Decision Boundary Comparison**

| Logistic Regression | L2-Norm Features | Degree-2 Polynomial Features |
|---|---|---|
| ![Logistic regression decision boundary](Media/rings_logistic_all.png) | ![L2-norm feature decision boundary](Media/rings_l2_norm_all.png) | ![Polynomial feature decision boundary](Media/rings_polynomial_degree2_all.png) |

**Test Set Predictions**

| Logistic Regression | Degree-2 Polynomial Features |
|---|---|
| ![Logistic regression test predictions](Media/rings_logistic_test.png) | ![Polynomial feature test predictions](Media/rings_polynomial_degree2_test.png) |

### Polynomial Regression

Polynomial models of different degrees are fitted to `poly_s.npz` and `poly_d.npz` to compare their fit and generalization.

**poly_s.npz**

![Original poly_s dataset](Media/poly_s_raw.png)

| Degree Comparison | Training vs Test MSE |
|---|---|
| ![poly_s degree comparison](Media/poly_s_degree_comparison.png) | ![poly_s MSE comparison](Media/poly_s_mse_comparison.png) |

**poly_d.npz**

![Original poly_d dataset](Media/poly_d_raw.png)

| Degree Comparison | Training vs Test MSE |
|---|---|
| ![poly_d degree comparison](Media/poly_d_degree_comparison.png) | ![poly_d MSE comparison](Media/poly_d_mse_comparison.png) |

**Degree-2 Training and Test Comparison (`poly_d.npz`)**

| Training | Test |
|---|---|
| ![poly_d degree 2 training fit](Media/poly_d_degree2_training.png) | ![poly_d degree 2 test fit](Media/poly_d_degree2_test.png) |

## Technologies

Python · NumPy · scikit-learn · Matplotlib
