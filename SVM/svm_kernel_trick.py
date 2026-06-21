import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------
# 1. PSEUDO-DATA GENERATION (2 Ellipses)
#------------------------------------------------------------------
d_1 = 100
d_2 = 100

def xy_gen(d_points, x0, y0, a, b):
    x_min, x_max = x0 - a, x0 + a
    x = np.random.uniform(x_min, x_max, d_points)
    
    y_min = y0 - b * np.sqrt(1.0 - ((x - x0) / a)**2)
    y_max = y0 + b * np.sqrt(1.0 - ((x - x0) / a)**2)
    y = np.random.uniform(y_min, y_max, d_points)
    
    return x, y

x1, y1 = xy_gen(d_1, x0=1, y0=2, a=1.5, b=1.0)
x2, y2 = xy_gen(d_2, x0=2, y0=3, a=3.0, b=1.8)

#--------------------------------------------------------------
# 2. TARGET AND DATASET PREPARATION
#--------------------------------------------------------------
# Target 1 belongs to Ellipse 1, Target 2 belongs to Ellipse 2. 
# We will map +1 to Ellipse 1 and -1 to Ellipse 2.
# We want to build training events as triplets (x, y, +1) and (x, y, -1).

# Organizing Ellipse 1 (Target +1)
# Create a column of 1s with the same length as x1
target1 = np.ones(len(x1))
Event1 = np.column_stack((x1, y1, target1))

# Organizing Ellipse 2 (Target -1)
# Create a column of -1s with the same length as x2
target2 = -np.ones(len(x2))
Event2 = np.column_stack((x2, y2, target2))

# Concatenating everything into a single dataset
dataset = np.vstack((Event1, Event2))

#--------------------------------------------------------------
# Taking 80% of the dataset for training
#--------------------------------------------------------------

# X contains (x, y) -> columns 0 and 1 (geometric positions)
# y contains the target -> column 2
X = dataset[:, :2] 
y = dataset[:, 2] 

# 80% train and 20% test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#--------------------------------------------------------------
# 3. KERNEL TRICK MANUAL (5D Mapping)
#--------------------------------------------------------------

# Basis transformation
def phi(x_vec):
    """ Maps [x, y] to a 5D space where quadratic terms become linear """
    x, y = x_vec[0], x_vec[1]
    return np.array([x, y, x**2, y**2, x*y])

def f(x_vec, w, b):
    """
    Represents f(x; w, b) = w · phi(x) + b
    x_vec: observation vector [x, y]
    w: weights found by the SVM [w1, w2, w3, w4, w5]
    b: bias (intercept)
    """
    return np.dot(w, phi(x_vec)) + b

#--------------------------------------------------------------
# Training, learning w and b
#--------------------------------------------------------------

# Transforming the original 2D training data to the 5D feature space
X_train_phi = np.array([phi(xi) for xi in X_train])

# 1. Creating and training the model (using linear kernel in 5D space)
clf_linear = svm.SVC(kernel='linear', C=1.0)
clf_linear.fit(X_train_phi, y_train)

# 2. Extracting optimized parameters
# w (5 components) is in coef_[0], bias b is in intercept_[0]
w = clf_linear.coef_[0]
b = clf_linear.intercept_[0]

print(f"Weights in 5D space (w): {w}")
print(f"Bias (b): {b:.4f}")

#--------------------------------------------------------------
# 4. FIGURES: VISUALIZATION OF NON-LINEAR BOUNDARY
#--------------------------------------------------------------

# Graph generator using the TEST set
plt.figure(figsize=(10, 8))

# Plotting test points (unseen data)
plt.scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], 
            c='royalblue', label='Test: Class +1 (Blue)', edgecolors='w', alpha=0.9, s=60)
plt.scatter(X_test[y_test==-1][:,0], X_test[y_test==-1][:,1], 
            c='darkorange', label='Test: Class -1 (Orange)', edgecolors='w', alpha=0.9, s=60)

# Creating the meshgrid to map the decision boundary f(x) = 0
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 150), np.linspace(ylim[0], ylim[1], 150))

# Calculating f for each point in the meshgrid projected in 5D space
Z = np.array([f(np.array([px, py]), w, b) for px, py in zip(np.ravel(xx), np.ravel(yy))])
Z = Z.reshape(xx.shape)

# Drawing the non-linear decision boundary (f=0) and support margins (f=1, f=-1)
contour = plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'red', 'gray'],
                      linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
plt.clabel(contour, inline=True, fontsize=10, fmt={-1:'Margin (-1)', 0:'Boundary (0)', 1:'Margin (+1)'})

# Graph aesthetics
plt.title('SVM: Non-Linear Boundary via $\phi(x)$ Transformation on Test Set', fontsize=14, fontweight='bold')
plt.xlabel('X Coordinate ($\lambda_1$)', fontsize=12)
plt.ylabel('Y Coordinate ($\lambda_2$)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.axis('equal')
plt.tight_layout()

# Saving the figure in the project folder
plt.savefig('outputs/plots/svm_kernel_boundary.png', dpi=300)

#--------------------------------------------------------------
# 5. MODEL EVALUATION METRICS
#--------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Transform the 2D test data into the 5D feature space using phi
X_test_phi = np.array([phi(xi) for xi in X_test])

# 2. Make predictions on the unseen test set
y_pred = clf_linear.predict(X_test_phi)

# 3. Calculate classification metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 4. Print the formatted results
print("\n" + "="*60)
print("             SVM EVALUATION METRICS (TEST SET)")
print("="*60)
print(f"1. Accuracy:  {acc:.4f}  (Overall correctness of the model)")
print(f"2. Precision: {prec:.4f}  (True Positives / Predicted Positives)")
print(f"3. Recall:    {rec:.4f}  (True Positives / Actual Positives)")
print(f"4. F1-Score:  {f1:.4f}  (Harmonic mean of Precision and Recall)")
print("-" * 60)
print("5. Confusion Matrix:")
print(f"   [{cm[0][0]:2d}  {cm[0][1]:2d}]  <-- Actual Class -1 (Orange)")
print(f"   [{cm[1][0]:2d}  {cm[1][1]:2d}]  <-- Actual Class +1 (Blue)")
print("     ^   ^")
print("     |   |")
print("     |   Predicted Class +1")
print("Predicted Class -1")
print("="*60 + "\n")