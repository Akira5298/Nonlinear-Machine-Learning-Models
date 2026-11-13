"""Polynomial regression experiments (Parts 2a/2b) with an interactive plot menu.

Preserves the original experiments, numeric summaries, four-degree comparisons,
and MSE analyses. Adds individually selectable training/test plots and PNG export.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DEGREES = (0, 1, 2, 5)
BASE = Path(__file__).resolve().parent


def find_dataset(filename):
    candidates = [BASE.parent / 'data' / filename, BASE / 'data' / filename,
                  BASE / filename, Path.cwd() / 'data' / filename,
                  Path.cwd() / filename]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Couldn't find {filename}. Place it in your project's data/ folder "
        f"or next to this script. Checked: {', '.join(map(str, candidates))}"
    )


def analyze_dataset(filename):
    # Parts 2(a) and 2(b): same calculations as the original, for each dataset.
    path = find_dataset(filename)
    with np.load(path) as data:
        X, Y = data['X'], data['Y']

    print(f"\n{'=' * 60}\nPOLYNOMIAL REGRESSION RESULTS FOR {filename}\n{'=' * 60}")
    print('Dataset shape:')
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim != 2 or X.shape[1] != 1 or Y.ndim != 2 or Y.shape[1] != 1:
        raise ValueError('Polynomial visualizations require one input and one output column.')

    print(f'Reshaped X shape: {X.shape}')
    print(f'Reshaped Y shape: {Y.shape}')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    print('\nData split:')
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')

    models, train_mses, test_mses = {}, {}, {}
    for degree in DEGREES:
        print(f'\nDegree {degree} Polynomial:')
        print('-' * 30)
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

        # Keep the original LinearRegression estimator; fitted polynomial curves
        # are produced by transforming x using the same polynomial feature map.
        model = LinearRegression()
        model.fit(X_train_poly, Y_train)
        Y_train_pred = model.predict(X_train_poly)
        Y_test_pred = model.predict(X_test_poly)
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        test_mse = mean_squared_error(Y_test, Y_test_pred)
        models[degree] = (model, poly_features)
        train_mses[degree] = train_mse
        test_mses[degree] = test_mse
        print(f'Training MSE: {train_mse:.6f}')
        print(f'Test MSE: {test_mse:.6f}')
        print('Model parameters:')
        print(f'  Intercept: {float(np.asarray(model.intercept_).item()):.6f}')
        if degree > 0:
            feature_names = poly_features.get_feature_names_out(['x'])
            for coef, name in zip(model.coef_.flatten(), feature_names):
                if name != '1':
                    print(f'  {name}: {coef:.6f}')

    print('\n' + '=' * 60)
    print(f'SUMMARY TABLE FOR {filename}')
    print('=' * 60)
    print('Degree | Train MSE  | Test MSE   | Difference')
    print('-' * 49)
    for degree in DEGREES:
        train_mse, test_mse = train_mses[degree], test_mses[degree]
        print(f'  {degree:<3}  | {train_mse:.6f} | {test_mse:.6f} | {test_mse - train_mse:.6f}')

    best_test_degree = min(DEGREES, key=lambda d: test_mses[d])
    print('\n' + '=' * 60)
    print(f'ANALYSIS FOR {filename}')
    print('=' * 60)
    print(f'Lowest test MSE achieved by degree {best_test_degree} polynomial: '
          f'{test_mses[best_test_degree]:.6f}')
    print('\nBased on the results:')
    print('- Degree 0 (constant): High error, underfitting')
    print(f"- Degree 1 (linear): {'Low' if test_mses[1] < 0.01 else 'High'} error")
    print(f"- Degree 2 (quadratic): {'Low' if test_mses[2] < 0.01 else 'Moderate'} error")
    print(f"- Degree 5: {'Low' if test_mses[5] < 0.01 else 'High'} error, potential overfitting")
    print(f'\nThe original polynomial degree was likely {best_test_degree} because:')
    print('1. It achieves the lowest test MSE')
    print('2. Visual inspection shows the best fit to the data pattern')
    print('3. Higher degrees may overfit, lower degrees may underfit')
    print('Note: These interpretations are heuristics, not proof of the true degree.')
    return dict(filename=filename, X=X, Y=Y, X_train=X_train, X_test=X_test,
                Y_train=Y_train, Y_test=Y_test, models=models,
                train_mses=train_mses, test_mses=test_mses)


def make_plot(info, kind, degree=None):
    name = Path(info['filename']).stem
    if kind == 'raw':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(info['X'].ravel(), info['Y'].ravel(), color='purple', alpha=0.7,
                   edgecolor='k')
        ax.set_title(f'Raw {info["filename"]} Dataset')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        file_name = f'{name}_raw.png'
    elif kind in ('training', 'test', 'combined'):
        if degree not in DEGREES:
            raise ValueError(f'Choose one of {DEGREES}')
        fig, ax = plt.subplots(figsize=(9, 6))
        model, poly_features = info['models'][degree]
        x_plot = np.linspace(info['X'].min(), info['X'].max(), 300).reshape(-1, 1)
        y_plot = model.predict(poly_features.transform(x_plot))
        if kind in ('training', 'combined'):
            ax.scatter(info['X_train'].ravel(), info['Y_train'].ravel(), alpha=0.6,
                       s=20, label='Training data', color='blue')
        if kind in ('test', 'combined'):
            ax.scatter(info['X_test'].ravel(), info['Y_test'].ravel(), alpha=0.6,
                       s=20, label='Test data', color='red')
        ax.plot(x_plot.ravel(), y_plot.ravel(), 'g-', linewidth=2,
                label=f'Degree {degree} fit')
        ax.set_title(f'{info["filename"]}: Degree {degree} ({kind.title()})\n'
                     f'Train MSE: {info["train_mses"][degree]:.4f}, '
                     f'Test MSE: {info["test_mses"][degree]:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        file_name = f'{name}_degree{degree}_{kind}.png'
    elif kind == 'comparison':
        # Original 2x2 plot, with all four polynomial degrees.
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        x_plot = np.linspace(info['X'].min(), info['X'].max(), 300).reshape(-1, 1)
        for degree, ax in zip(DEGREES, axes.flat):
            model, poly = info['models'][degree]
            y_plot = model.predict(poly.transform(x_plot))
            ax.scatter(info['X_train'].ravel(), info['Y_train'].ravel(), alpha=0.6,
                       s=20, label='Training data', color='blue')
            ax.scatter(info['X_test'].ravel(), info['Y_test'].ravel(), alpha=0.6,
                       s=20, label='Test data', color='red')
            ax.plot(x_plot.ravel(), y_plot.ravel(), 'g-', linewidth=2,
                    label=f'Degree {degree} fit')
            ax.set_title(f'Degree {degree} Polynomial ({name})\n'
                         f'Train MSE: {info["train_mses"][degree]:.4f}, '
                         f'Test MSE: {info["test_mses"][degree]:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        file_name = f'{name}_degree_comparison.png'
    elif kind == 'mse':
        # Original training vs test MSE plot with log scale.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(DEGREES, [info['train_mses'][d] for d in DEGREES], 'bo-',
                linewidth=2, markersize=8, label='Training MSE')
        ax.plot(DEGREES, [info['test_mses'][d] for d in DEGREES], 'ro-',
                linewidth=2, markersize=8, label='Test MSE')
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title(f'Training vs Test MSE by Polynomial Degree ({name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if all(info[k][d] > 0 for k in ('train_mses', 'test_mses') for d in DEGREES):
            ax.set_yscale('log')
        file_name = f'{name}_mse_comparison.png'
    else:
        raise ValueError(f'Unrecognized plot type: {kind}')

    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title(file_name.replace('.png', ''))
    except (AttributeError, NotImplementedError):
        pass
    return fig, file_name


def show_plots(info, requests):
    save = input('Save selected image(s) as PNG? (y/n): ').strip().lower() == 'y'
    for kind, degree in requests:
        fig, file_name = make_plot(info, kind, degree)
        if save:
            output = BASE / 'plots'
            output.mkdir(parents=True, exist_ok=True)
            path = output / file_name
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.show()  # On some backends, close the figure to proceed.
        plt.close(fig)


def select_degree():
    value = input('Choose polynomial degree (0, 1, 2, 5): ').strip()
    try:
        degree = int(value)
    except ValueError:
        degree = None
    if degree not in DEGREES:
        print('Invalid degree.')
        return None
    return degree


def dataset_menu(info):
    while True:
        print(f'\n=== {info["filename"]} IMAGE MENU ===')
        print('1. Raw dataset')
        print('2. Training graph (choose degree)')
        print('3. Test graph (choose degree)')
        print('4. Both training and test graphs (separate images)')
        print('5. Original combined graph (choose degree)')
        print('6. Original 2x2 degree comparison')
        print('7. Original MSE vs degree comparison')
        print('8. All images for this dataset')
        print('0. Back')
        choice = input('Select an option: ').strip()
        if choice == '0':
            return
        if choice == '1':
            requests = [('raw', None)]
        elif choice in ('2', '3', '4', '5'):
            degree = select_degree()
            if degree is None:
                continue
            if choice == '4':
                requests = [('training', degree), ('test', degree)]
            else:
                requests = [({'2': 'training', '3': 'test', '5': 'combined'}[choice], degree)]
        elif choice == '6':
            requests = [('comparison', None)]
        elif choice == '7':
            requests = [('mse', None)]
        elif choice == '8':
            requests = ([('raw', None)] +
                        [(kind, degree) for degree in DEGREES
                         for kind in ('training', 'test', 'combined')] +
                        [('comparison', None), ('mse', None)])
        else:
            print('Invalid option.')
            continue
        show_plots(info, requests)


def main():
    cache = {}
    while True:
        print('\n=== POLYNOMIAL REGRESSION ANALYZER ===')
        print('1. Part 2(a): poly_s.npz')
        print('2. Part 2(b): poly_d.npz')
        print('3. Both datasets (run analyses, then choose graphs)')
        print('0. Exit')
        choice = input('Select a dataset: ').strip()
        if choice == '0':
            print('Exiting...')
            break
        filenames = {'1': ['poly_s.npz'], '2': ['poly_d.npz'],
                     '3': ['poly_s.npz', 'poly_d.npz']}.get(choice)
        if filenames is None:
            print('Invalid option.')
            continue
        for filename in filenames:
            try:
                if filename not in cache:
                    cache[filename] = analyze_dataset(filename)
                dataset_menu(cache[filename])
            except (FileNotFoundError, KeyError, ValueError) as error:
                print(f'Error with {filename}: {error}')


if __name__ == '__main__':
    main()

