import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Existing Part 1(a)-(d) experiments are preserved below.
# New: interactive graph selection, separate train/test graphs and PNG export.

BASE = Path(__file__).resolve().parent
PLOTS_DIR = BASE / 'plots'


def load_data():
    # Work from the repository root, src/, or alongside this script.
    candidates = [BASE.parent / 'data' / 'rings.npz', BASE / 'data' / 'rings.npz',
                  BASE / 'rings.npz', Path.cwd() / 'rings.npz',
                  Path.cwd() / 'data' / 'rings.npz']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('rings.npz not found. Place it in data/ or beside this script.')
    with np.load(path) as data:
        X, Y = data['X'], data['Y']
    return X, Y.reshape(-1, 1)


def plot_data(X, Y, ax):
    # Original markers and class labels preserved.
    Y = np.asarray(Y).reshape(-1, 1)
    positive_samples = X[Y[:, 0] == 1, :]
    negative_samples = X[Y[:, 0] == -1, :]
    ax.plot(positive_samples[:, 0], positive_samples[:, 1], 'x', label='$+1$')
    ax.plot(negative_samples[:, 0], negative_samples[:, 1], 'o', label='$-1$')


def visualize_raw(X, Y):
    """Part 1(a): Visualizes only the dataset without separator."""
    fig, ax = plt.subplots()
    plot_data(X, Y, ax)
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_title('Rings Dataset')
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def visualize_nonlin(fun, X, Y, resolution=50):
    """Original part 1(b-d) visualization with zero-level decision separator."""
    fig, ax = plt.subplots()
    plot_data(X, Y, ax)
    x = np.linspace(-5, 5, resolution)
    xx, yy = np.meshgrid(x, x)
    grid = np.c_[xx.flatten(), yy.flatten()]
    decisions = fun(grid)
    ax.contour(xx, yy, decisions.reshape(resolution, resolution),
               levels=np.array([0]), colors='red', linewidths=2)
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def run_original_analyses(X, Y):
    """Compute all original experiment features, models, and errors once."""
    # //////////////////////////// part 1(b) ////////////////////////////
    Y_flat = Y.flatten()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, Y_train)
    Y_train_pred = lr_model.predict(X_train)
    Y_test_pred = lr_model.predict(X_test)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy
    print(f'Training Error: {train_error:.4f}')
    print(f'Test Error: {test_error:.4f}')

    def logistic_hypothesis(x):
        return lr_model.decision_function(x)

    # //////////////////////////// part 1(c) ////////////////////////////
    X_l2_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_with_l2 = np.hstack([X, X_l2_norm])
    print(f'Original feature shape: {X.shape}')
    print(f'With L2 norm feature shape: {X_with_l2.shape}')
    X_train_l2, X_test_l2, Y_train_l2, Y_test_l2 = train_test_split(
        X_with_l2, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)
    lr_model_l2 = LogisticRegression(random_state=42, max_iter=1000)
    lr_model_l2.fit(X_train_l2, Y_train_l2)
    Y_train_pred_l2 = lr_model_l2.predict(X_train_l2)
    Y_test_pred_l2 = lr_model_l2.predict(X_test_l2)
    train_accuracy_l2 = accuracy_score(Y_train_l2, Y_train_pred_l2)
    test_accuracy_l2 = accuracy_score(Y_test_l2, Y_test_pred_l2)
    train_error_l2 = 1 - train_accuracy_l2
    test_error_l2 = 1 - test_accuracy_l2
    print(f'Training Error (with L2 norm): {train_error_l2:.4f}')
    print(f'Test Error (with L2 norm): {test_error_l2:.4f}')

    def logistic_hypothesis_l2(x):
        x_l2_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x_with_l2 = np.hstack([x, x_l2_norm])
        return lr_model_l2.decision_function(x_with_l2)

    # //////////////////////////// part 1(d) ////////////////////////////
    x1_squared = X[:, 0]**2
    x1_x2 = X[:, 0] * X[:, 1]
    x2_squared = X[:, 1]**2
    X_poly = np.column_stack([X, x1_squared, x1_x2, x2_squared])
    print(f'Original feature shape: {X.shape}')
    print(f'With polynomial features shape: {X_poly.shape}')
    X_train_poly, X_test_poly, Y_train_poly, Y_test_poly = train_test_split(
        X_poly, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_poly, Y_train_poly)
    prediction = clf.predict(X_test_poly)
    train_accuracy_poly = accuracy_score(Y_train_poly, clf.predict(X_train_poly))
    test_accuracy_poly = accuracy_score(Y_test_poly, prediction)
    train_error_poly = 1 - train_accuracy_poly
    test_error_poly = 1 - test_accuracy_poly
    print(f'Training Error (with polynomial features): {train_error_poly:.4f}')
    print(f'Test Error (with polynomial features): {test_error_poly:.4f}')

    def logistic_hypothesis_poly(x):
        x1_sq = x[:, 0]**2
        x1_x2_prod = x[:, 0] * x[:, 1]
        x2_sq = x[:, 1]**2
        x_poly = np.column_stack([x, x1_sq, x1_x2_prod, x2_sq])
        return clf.decision_function(x_poly)

    return {
        'raw': (None, 'Rings Dataset', X, Y, None, None, 'rings_raw'),
        'logistic': (logistic_hypothesis, 'Logistic Regression Decision Boundary',
                     X_train, Y_train, X_test, Y_test, 'rings_logistic'),
        'l2_norm': (logistic_hypothesis_l2, 'Linear Classifier with L2 Norm Feature',
                    X_train, Y_train, X_test, Y_test, 'rings_l2_norm'),
        'polynomial_degree2': (logistic_hypothesis_poly,
                               'Linear Classifier with Polynomial Features (Degree 2)',
                               X_train, Y_train, X_test, Y_test,
                               'rings_polynomial_degree2')
    }, {
        'logistic': (train_error, test_error),
        'l2_norm': (train_error_l2, test_error_l2),
        'polynomial_degree2': (train_error_poly, test_error_poly)
    }


def show_graph(key, split, experiments, errors, save=False):
    hypothesis, title, X_train, Y_train, X_test, Y_test, stem = experiments[key]
    if key == 'raw':
        fig, ax = visualize_raw(X_train, Y_train)
        image_name = f'{stem}.png'
        window_name = 'rings.npz - Raw Dataset'
    else:
        if split == 'training':
            X_display, Y_display = X_train, Y_train
        elif split == 'test':
            X_display, Y_display = X_test, Y_test
        elif split == 'all':
            # Original graph: all samples together, as in the original script.
            X_display, Y_display = experiments['raw'][2:4]
        else:
            raise ValueError('split must be training, test, or all')
        fig, ax = visualize_nonlin(hypothesis, X_display, Y_display)
        train_error, test_error = errors[key]
        ax.set_title(f'{title} ({split.title()})\n'
                     f'Train Error: {train_error:.3f}, Test Error: {test_error:.3f}')
        image_name = f'{stem}_{split}.png'
        window_name = f'{stem} - {split.title()}'

    # Different titles in graphical backends, rather than the generic Figure 1.
    try:
        fig.canvas.manager.set_window_title(window_name)
    except (AttributeError, NotImplementedError):
        pass
    if save:
        PLOTS_DIR.mkdir(exist_ok=True)
        destination = PLOTS_DIR / image_name
        fig.savefig(destination, dpi=300, bbox_inches='tight')
        print(f'Saved: {destination}')
    plt.show()
    plt.close(fig)


def ask_save():
    return input('Save selected image(s) as PNG? (y/n): ').strip().lower() in ('y', 'yes')


def main():
    X, Y = load_data()
    print('\n=== Original Nonlinear Classification Analyses ===')
    experiments, errors = run_original_analyses(X, Y)
    experiment_options = {'1': 'raw', '2': 'logistic', '3': 'l2_norm',
                          '4': 'polynomial_degree2'}

    while True:
        print('\n=== NONLINEAR CLASSIFICATION ANALYZER ===')
        print('1. Part 1(a): Raw rings dataset')
        print('2. Part 1(b): Logistic regression')
        print('3. Part 1(c): Logistic regression + L2 norm feature')
        print('4. Part 1(d): Logistic regression + degree-2 features')
        print('5. All graphs (raw + training/test + original combined graphs)')
        print('0. Exit')
        choice = input('Select experiment (0-5): ').strip()
        if choice == '0':
            break
        if choice not in (*experiment_options, '5'):
            print('Invalid option.')
            continue
        if choice == '5':
            selected = [('raw', None)] + [(k, s) for k in
                ('logistic', 'l2_norm', 'polynomial_degree2') for s in ('training', 'test', 'all')]
        else:
            key = experiment_options[choice]
            if key == 'raw':
                selected = [(key, None)]
            else:
                print('\n1. Training graph\n2. Test graph\n3. Both separate graphs')
                print('4. Original graph (all samples combined)\n0. Back')
                split_choice = input('Choose graph (0-4): ').strip()
                if split_choice == '0':
                    continue
                split_options = {'1': ['training'], '2': ['test'],
                                 '3': ['training', 'test'], '4': ['all']}
                if split_choice not in split_options:
                    print('Invalid option.')
                    continue
                selected = [(key, split) for split in split_options[split_choice]]
        save = ask_save()
        for key, split in selected:
            show_graph(key, split, experiments, errors, save=save)


if __name__ == '__main__':
    main()

