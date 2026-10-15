import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

#////////////////////////////////part2 a/////////////////////////

# Load the dataset
data = np.load('poly_s.npz')
X = data['X']  # Features
Y = data['Y']  # Labels

print("Dataset shape:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Reshape X to be 2D if it's 1D
if X.ndim == 1:
    X = X.reshape(-1, 1)
if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

print(f"Reshaped X shape: {X.shape}")
print(f"Reshaped Y shape: {Y.shape}")

# Visualize only the raw poly_s.npz dataset
plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='purple', alpha=0.7, edgecolor='k')
plt.title('Raw poly_s.npz Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(f"\nData split:")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Polynomial degrees to test
degrees = [0, 1, 2, 5]
models = {}
train_mses = {}
test_mses = {}

print("\n" + "="*60)
print("POLYNOMIAL REGRESSION RESULTS")
print("="*60)

# Train polynomial regression for each degree
for degree in degrees:
    print(f"\nDegree {degree} Polynomial:")
    print("-" * 30)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    
    # Make predictions
    Y_train_pred = model.predict(X_train_poly)
    Y_test_pred = model.predict(X_test_poly)
    
    # Calculate MSEs
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    
    # Store results
    models[degree] = (model, poly_features)
    train_mses[degree] = train_mse
    test_mses[degree] = test_mse
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Print model parameters
    coefficients = model.coef_.flatten()
    intercept = model.intercept_
    
    print(f"Model parameters:")
    print(f"  Intercept: {intercept.item():.6f}")
    if degree > 0:
        feature_names = poly_features.get_feature_names_out(['x'])
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            if name != '1':  # Skip the bias term (already shown as intercept)
                print(f"  {name}: {coef:.6f}")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print("Degree | Train MSE  | Test MSE   | Difference")
print("-" * 45)
for degree in degrees:
    train_mse = train_mses[degree]
    test_mse = test_mses[degree]
    diff = test_mse - train_mse
    print(f"  {degree}    | {train_mse:.6f} | {test_mse:.6f} | {diff:.6f}")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
axes = [ax1, ax2, ax3, ax4]

# Sort data for smooth plotting
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

for i, degree in enumerate(degrees):
    ax = axes[i]
    model, poly_features = models[degree]
    
    # Transform plotting data
    X_plot_poly = poly_features.transform(X_plot)
    Y_plot_pred = model.predict(X_plot_poly)
    
    # Plot training data
    ax.scatter(X_train, Y_train, alpha=0.6, s=20, label='Training data', color='blue')
    # Plot test data
    ax.scatter(X_test, Y_test, alpha=0.6, s=20, label='Test data', color='red')
    # Plot fitted curve
    ax.plot(X_plot, Y_plot_pred, 'g-', linewidth=2, label=f'Degree {degree} fit')
    
    ax.set_title(f'Degree {degree} Polynomial\nTrain MSE: {train_mses[degree]:.4f}, Test MSE: {test_mses[degree]:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis plot: MSE vs Degree
fig, ax = plt.subplots(figsize=(10, 6))
degrees_list = list(degrees)
train_mse_list = [train_mses[d] for d in degrees_list]
test_mse_list = [test_mses[d] for d in degrees_list]

ax.plot(degrees_list, train_mse_list, 'bo-', label='Training MSE', linewidth=2, markersize=8)
ax.plot(degrees_list, test_mse_list, 'ro-', label='Test MSE', linewidth=2, markersize=8)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Training vs Test MSE by Polynomial Degree')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale to better see differences
plt.tight_layout()
plt.show()

# Analysis and conclusion
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

best_test_degree = min(degrees, key=lambda d: test_mses[d])
print(f"Lowest test MSE achieved by degree {best_test_degree} polynomial: {test_mses[best_test_degree]:.6f}")

print(f"\nBased on the results:")
print(f"- Degree 0 (constant): High error, underfitting")
print(f"- Degree 1 (linear): {'Low' if test_mses[1] < 0.01 else 'High'} error")
print(f"- Degree 2 (quadratic): {'Low' if test_mses[2] < 0.01 else 'Moderate'} error")
print(f"- Degree 5: {'Low' if test_mses[5] < 0.01 else 'High'} error, potential overfitting")

print(f"\nThe original polynomial degree was likely {best_test_degree} because:")
print(f"1. It achieves the lowest test MSE")
print(f"2. Visual inspection shows the best fit to the data pattern")
print(f"3. Higher degrees may overfit, lower degrees may underfit")

#////////////////////////////////part2 b/////////////////////////

# Load the dataset - CHANGED FROM poly_s.npz to poly_d.npz
data_d = np.load('poly_d.npz')
X_d = data_d['X']  # Features
Y_d = data_d['Y']  # Labels

print("Dataset shape:")
print(f"X_d shape: {X_d.shape}")
print(f"Y_d shape: {Y_d.shape}")

# Reshape X to be 2D if it's 1D
if X_d.ndim == 1:
    X_d = X_d.reshape(-1, 1)
if Y_d.ndim == 1:
    Y_d = Y_d.reshape(-1, 1)

print(f"Reshaped X_d shape: {X_d.shape}")
print(f"Reshaped Y_d shape: {Y_d.shape}")

# Visualize only the raw poly_d.npz dataset - UPDATED TITLE
plt.figure(figsize=(8,6))
plt.scatter(X_d, Y_d, color='purple', alpha=0.7, edgecolor='k')
plt.title('Raw poly_d.npz Dataset')  # Updated title
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()

# Split the data into training and test sets
X_d_train, X_d_test, Y_d_train, Y_d_test = train_test_split(X_d, Y_d, test_size=0.3, random_state=42)

print(f"\nData split:")
print(f"Training set size: {X_d_train.shape[0]}")
print(f"Test set size: {X_d_test.shape[0]}")

# Polynomial degrees to test
degrees_d = [0, 1, 2, 5]
models_d = {}
train_mses_d = {}
test_mses_d = {}

print("\n" + "="*60)
print("POLYNOMIAL REGRESSION RESULTS FOR POLY_D")
print("="*60)

# Train polynomial regression for each degree
for degree in degrees_d:
    print(f"\nDegree {degree} Polynomial:")
    print("-" * 30)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_d_train_poly = poly_features.fit_transform(X_d_train)
    X_d_test_poly = poly_features.transform(X_d_test)
    
    # Train linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_d_train_poly, Y_d_train)
    
    # Make predictions
    Y_d_train_pred = model.predict(X_d_train_poly)
    Y_d_test_pred = model.predict(X_d_test_poly)
    
    # Calculate MSEs
    train_mse = mean_squared_error(Y_d_train, Y_d_train_pred)
    test_mse = mean_squared_error(Y_d_test, Y_d_test_pred)
    
    # Store results
    models_d[degree] = (model, poly_features)
    train_mses_d[degree] = train_mse
    test_mses_d[degree] = test_mse
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Print model parameters
    coefficients = model.coef_.flatten()
    intercept = model.intercept_
    
    print(f"Model parameters:")
    print(f"  Intercept: {intercept.item():.6f}")
    if degree > 0:
        feature_names = poly_features.get_feature_names_out(['x'])
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            if name != '1':  # Skip the bias term (already shown as intercept)
                print(f"  {name}: {coef:.6f}")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE FOR POLY_D")
print("="*60)
print("Degree | Train MSE  | Test MSE   | Difference")
print("-" * 45)
for degree in degrees_d:
    train_mse = train_mses_d[degree]
    test_mse = test_mses_d[degree]
    diff = test_mse - train_mse
    print(f"  {degree}    | {train_mse:.6f} | {test_mse:.6f} | {diff:.6f}")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
axes = [ax1, ax2, ax3, ax4]

# Sort data for smooth plotting
X_d_plot = np.linspace(X_d.min(), X_d.max(), 300).reshape(-1, 1)

for i, degree in enumerate(degrees_d):
    ax = axes[i]
    model, poly_features = models_d[degree]
    
    # Transform plotting data
    X_d_plot_poly = poly_features.transform(X_d_plot)
    Y_d_plot_pred = model.predict(X_d_plot_poly)
    
    # Plot training data
    ax.scatter(X_d_train, Y_d_train, alpha=0.6, s=20, label='Training data', color='blue')
    # Plot test data
    ax.scatter(X_d_test, Y_d_test, alpha=0.6, s=20, label='Test data', color='red')
    # Plot fitted curve
    ax.plot(X_d_plot, Y_d_plot_pred, 'g-', linewidth=2, label=f'Degree {degree} fit')
    
    ax.set_title(f'Degree {degree} Polynomial (poly_d)\nTrain MSE: {train_mses_d[degree]:.4f}, Test MSE: {test_mses_d[degree]:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis plot: MSE vs Degree
fig, ax = plt.subplots(figsize=(10, 6))
degrees_d_list = list(degrees_d)
train_mse_d_list = [train_mses_d[d] for d in degrees_d_list]
test_mse_d_list = [test_mses_d[d] for d in degrees_d_list]

ax.plot(degrees_d_list, train_mse_d_list, 'bo-', label='Training MSE', linewidth=2, markersize=8)
ax.plot(degrees_d_list, test_mse_d_list, 'ro-', label='Test MSE', linewidth=2, markersize=8)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Training vs Test MSE by Polynomial Degree (poly_d)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale to better see differences
plt.tight_layout()
plt.show()

# Analysis and conclusion
print("\n" + "="*60)
print("ANALYSIS FOR POLY_D")
print("="*60)

best_test_degree_d = min(degrees_d, key=lambda d: test_mses_d[d])
print(f"Lowest test MSE achieved by degree {best_test_degree_d} polynomial: {test_mses_d[best_test_degree_d]:.6f}")

print(f"\nBased on the results:")
print(f"- Degree 0 (constant): High error, underfitting")
print(f"- Degree 1 (linear): {'Low' if test_mses_d[1] < 0.01 else 'High'} error")
print(f"- Degree 2 (quadratic): {'Low' if test_mses_d[2] < 0.01 else 'Moderate'} error")
print(f"- Degree 5: {'Low' if test_mses_d[5] < 0.01 else 'High'} error, potential overfitting")

print(f"\nThe original polynomial degree was likely {best_test_degree_d} because:")
print(f"1. It achieves the lowest test MSE")
print(f"2. Visual inspection shows the best fit to the data pattern")
print(f"3. Higher degrees may overfit, lower degrees may underfit")
