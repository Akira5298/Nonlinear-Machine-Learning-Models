import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


#////////////////////////////////part1 a/////////////////////////

# Load the dataset
data = np.load('rings.npz')
X = data['X']  # Features
Y = data['Y']  # Labels

def plot_data(X, Y, ax):
    positive_samples = X[Y[:, 0] == 1, :]
    negative_samples = X[Y[:, 0] == -1, :]
    ax.plot(positive_samples[:, 0], positive_samples[:, 1], 'x', label='$+1$')
    ax.plot(negative_samples[:, 0], negative_samples[:, 1], 'o', label='$-1$')

def visualize_nonlin(X, Y):
    """
    Visualizes only the dataset without separator.
    """
    fig, ax = plt.subplots()
    
    # Plot data only
    plot_data(X, Y, ax)
    
    # Annotations
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_title('Rings Dataset')
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

# Visualize data only
fig, ax = visualize_nonlin(X, Y)

# Show the plot
plt.show()

#/////////////////////////////////part1 b//////////////////////////////////////

def visualize_nonlin(fun, X, Y, resolution=50):
    """
    Visualizes the dataset together with the separator
    """
    fig, ax = plt.subplots()
    # Data
    plot_data(X, Y, ax)
    # Separator
    x = np.linspace(-5, 5, resolution)
    xx, yy = np.meshgrid(x, x)
    grid = np.c_[xx.flatten(), yy.flatten()]
    decisions = fun(grid)
    ax.contour(
        xx, yy, decisions.reshape(resolution, resolution), levels=np.array([0]), colors='red', linewidths=2)
    # Annotations
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
    
# Add this code after your basic plotting
# Split the data into training and test sets
Y_flat = Y.flatten()  # Flatten Y for sklearn compatibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Logistic Regression classifier
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, Y_train)

# Make predictions
Y_train_pred = lr_model.predict(X_train)
Y_test_pred = lr_model.predict(X_test)

# Calculate training and test errors
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

train_error = 1 - train_accuracy
test_error = 1 - test_accuracy

print(f"Training Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")

# Create hypothesis function for visualization
def logistic_hypothesis(x):
    return lr_model.decision_function(x)

# Visualize the decision boundary
fig, ax = visualize_nonlin(logistic_hypothesis, X, Y)
ax.set_title(f'Logistic Regression Decision Boundary\nTrain Error: {train_error:.3f}, Test Error: {test_error:.3f}')
plt.show()

#////////////////////////////////part1 c/////////////////////////

# Add L2 norm as a new feature
X_l2_norm = np.linalg.norm(X, axis=1, keepdims=True)  # ||x||2 = sqrt(x1^2 + x2^2)
X_with_l2 = np.hstack([X, X_l2_norm])  # Combine original features with L2 norm

print(f"Original feature shape: {X.shape}")
print(f"With L2 norm feature shape: {X_with_l2.shape}")

# Split data with new features
X_train_l2, X_test_l2, Y_train_l2, Y_test_l2 = train_test_split(X_with_l2, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)

# Train logistic regression with L2 norm feature
lr_model_l2 = LogisticRegression(random_state=42, max_iter=1000)
lr_model_l2.fit(X_train_l2, Y_train_l2)

# Make predictions
Y_train_pred_l2 = lr_model_l2.predict(X_train_l2)
Y_test_pred_l2 = lr_model_l2.predict(X_test_l2)

# Calculate errors
train_accuracy_l2 = accuracy_score(Y_train_l2, Y_train_pred_l2)
test_accuracy_l2 = accuracy_score(Y_test_l2, Y_test_pred_l2)
train_error_l2 = 1 - train_accuracy_l2
test_error_l2 = 1 - test_accuracy_l2

print(f"Training Error (with L2 norm): {train_error_l2:.4f}")
print(f"Test Error (with L2 norm): {test_error_l2:.4f}")

# Create hypothesis function for decision boundary (using original 2D space)
def logistic_hypothesis_l2(x):
    # Add L2 norm feature to the 2D grid points
    x_l2_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_with_l2 = np.hstack([x, x_l2_norm])
    return lr_model_l2.decision_function(x_with_l2)

# Visualize decision boundary
fig, ax = visualize_nonlin(logistic_hypothesis_l2, X, Y)
ax.set_title(f'Linear Classifier with L2 Norm Feature\nTrain Error: {train_error_l2:.3f}, Test Error: {test_error_l2:.3f}')
plt.show()

#////////////////////////////////part1 d/////////////////////////

# Part D: Classification with degree 2 polynomial features
from sklearn.linear_model import LogisticRegression

# Create degree 2 polynomial features: x1^2, x1*x2, x2^2
x1_squared = X[:, 0]**2
x1_x2 = X[:, 0] * X[:, 1]
x2_squared = X[:, 1]**2

# Combine original features with polynomial features
X_poly = np.column_stack([X, x1_squared, x1_x2, x2_squared])

print(f"Original feature shape: {X.shape}")
print(f"With polynomial features shape: {X_poly.shape}")

# Split data with polynomial features
X_train_poly, X_test_poly, Y_train_poly, Y_test_poly = train_test_split(X_poly, Y_flat, test_size=0.3, random_state=42, stratify=Y_flat)

# Train classifier
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_poly, Y_train_poly)

# Make predictions
prediction = clf.predict(X_test_poly)

# Calculate errors
train_accuracy_poly = accuracy_score(Y_train_poly, clf.predict(X_train_poly))
test_accuracy_poly = accuracy_score(Y_test_poly, prediction)
train_error_poly = 1 - train_accuracy_poly
test_error_poly = 1 - test_accuracy_poly

print(f"Training Error (with polynomial features): {train_error_poly:.4f}")
print(f"Test Error (with polynomial features): {test_error_poly:.4f}")

# Create hypothesis function for decision boundary
def logistic_hypothesis_poly(x):
    # Create polynomial features for the grid points
    x1_sq = x[:, 0]**2
    x1_x2_prod = x[:, 0] * x[:, 1]
    x2_sq = x[:, 1]**2
    x_poly = np.column_stack([x, x1_sq, x1_x2_prod, x2_sq])
    return clf.decision_function(x_poly)

# Visualize decision boundary
fig, ax = visualize_nonlin(logistic_hypothesis_poly, X, Y)
ax.set_title(f'Linear Classifier with Polynomial Features (Degree 2)\nTrain Error: {train_error_poly:.3f}, Test Error: {test_error_poly:.3f}')
plt.show()

#////////////////////////////////part2 a/////////////////////////



