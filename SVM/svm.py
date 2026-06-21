import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

#-------------------------------------
# We have N = 2 characteristics, namely, x and y coordinate
N = 2 

#------------------------------------------------------------------
# PSEUDO-DATA GENERATION
#------------------------------------------------------------------
# Pseudo-data generation - we generate d_points random numbers in [x0 - a, x0 + a] through an uniform distribution

# Number os random points generated in each elipse
d_1 = 100
d_2 = 100

#Sample number
d = d_1 + d_2

def xy_gen(d_points, x0, y0, a, b):

    x_min = x0 - a
    x_max = x0 + a

    # np.random.uniform(limite_inferior, limite_superior, quantidade)
    x = np.random.uniform(x_min, x_max, d_points)

    y_min = y0 - b*np.sqrt(1.0 - ((x-x0)/a)**2)
    y_max = y0 + b*np.sqrt(1.0 - ((x-x0) /a)**2)

    y = np.random.uniform(y_min, y_max, d_points)

    return (x, y)

#-------------------------------------
# Graph generator

# 1. Generating the pseudo-data for 2 ellipses (Binary Classification)
x1, y1 = xy_gen(d_1, x0=1, y0=2, a=1.5, b=1.0)
c1 = np.full(d_1, 'royalblue')

x2, y2 = xy_gen(d_2, x0=2, y0=3, a=3.0, b=1.8)
c2 = np.full(d_2, 'darkorange')

#-------------------------------------
# We create the X: all points in a single matrix for SVM
x_svm = np.concatenate([x1, x2])
y_svm = np.concatenate([y1, y2])
c_svm = np.concatenate([c1, c2])

# Matriz X com 200 pontos (100 de cada classe)
X = np.column_stack((x_svm, y_svm)) 

colors_labeled = c_svm.tolist()

# Class labels:
# +1 for the firts ellipse (Azul)
# -1 for the second ellipse (Laranja)
y_labels = np.array([1] * len(x1) + [-1] * len(x2))

# ============================================================
# 2. SVM COST FUNCTION
# ============================================================
def svm_cost(X, y, w, b, lambda_reg=0.01):
    """
    We calculate the cost function of the linear SVM:

        J(w,b) = (lambda/2) ||w||^2 + média da hinge loss

    onde:
        hinge_i = max(0, 1 - y_i (w·x_i + b))
    """
    scores = np.dot(X, w) + b
    margins = y * scores
    hinge_losses = np.maximum(0.0, 1.0 - margins)

    regularization = 0.5 * lambda_reg * np.dot(w, w)
    data_loss = np.mean(hinge_losses)

    return regularization + data_loss


# ============================================================
# 3. TRAINS OF SVM LINEAR
# ============================================================
w = np.zeros(2)  
b = 0.0           

# Hiperparams
learning_rate = 0.01
epochs = 1000
lambda_reg = 0.01

loss_history = []

for epoch in range(epochs):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y_labels[indices]

    # SGD - Stochastic Gradient Descent
    for x_i, y_i in zip(X_shuffled, y_shuffled):
        margin = y_i * (np.dot(w, x_i) + b)

        if margin >= 1:
            w = w - learning_rate * (lambda_reg * w)
        else:
            w = w - learning_rate * (lambda_reg * w - y_i * x_i)
            b = b + learning_rate * y_i

    current_loss = svm_cost(X, y_labels, w, b, lambda_reg=lambda_reg)
    loss_history.append(current_loss)

print(f"Pesos finais: w = {w}")
print(f"Bias final: b = {b:.4f}")
print(f"Loss final: {loss_history[-1]:.4f}")

#-------------------------------------------------------------------
# FIGURES
#-------------------------------------------------------------------

# 2. Figure configuration
plt.figure(figsize=(8, 6))

# 3. Plot. The scatter don't join the points with a line as the plot. 
plt.scatter(x1, y1, color=c1[0], alpha=1, edgecolors='w', linewidth=0.5, label='Classe +1')
plt.scatter(x2, y2, color=c2[0], alpha=1, edgecolors='w', linewidth=0.5, label='Classe -1')

# 4. Labels
plt.title('Synthetic Data Generation (2 Ellipses)', fontsize=14, fontweight='bold')
plt.xlabel('X coordinate', fontsize=12)
plt.ylabel('Y coordinate', fontsize=12)

# Axis x and y have the same scale
plt.axis('equal')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 5. Save and show the graph
plt.tight_layout()
plt.savefig('outputs/plots/pseudo_data_ellipses.png', dpi=300)

# ============================================================
# 4. FRONTIER OF DECISION
# ============================================================
plt.figure(figsize=(10, 8))

# Points of class +1
plt.scatter(
    x1, y1,
    color="royalblue",
    edgecolors="w",
    linewidth=0.5,
    s=60,
    label="Classe +1"
)

# Points of class -1
plt.scatter(
    x2, y2,
    color="darkorange",
    edgecolors="w",
    linewidth=0.5,
    s=60,
    label="Classe -1"
)

x_plot = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 300)

# Frontier: w0*x + w1*y + b = 0
# y = -(w0*x + b)/w1
if abs(w[1]) > 1e-12:
    y_decision = -(w[0] * x_plot + b) / w[1]
    plt.plot(
        x_plot, y_decision,
        color="red",
        linewidth=2.5,
        label="Fronteira de decisão"
    )

    # w·x + b = +1 e w·x + b = -1
    y_margin_pos = (1.0 - w[0] * x_plot - b) / w[1]
    y_margin_neg = (-1.0 - w[0] * x_plot - b) / w[1]

    plt.plot(x_plot, y_margin_pos, "k--", alpha=0.5, label="Margens")
    plt.plot(x_plot, y_margin_neg, "k--", alpha=0.5)

plt.title("SVM Linear (Implementação Manual)", fontsize=15, fontweight="bold")
plt.xlabel("Coordenada x", fontsize=12)
plt.ylabel("Coordenada y", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.axis("equal")
plt.tight_layout()

plt.savefig('outputs/plots/svm_decision_boundary.png', dpi=300)

# ============================================================
# 5. COST FUNCTION EVOLUTION
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='purple', linewidth=2)
plt.title("Evolução da Função de Custo (Hinge Loss + Regularização)", fontsize=14, fontweight="bold")
plt.xlabel("Época", fontsize=12)
plt.ylabel("Loss (Custo)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()    

plt.savefig('outputs/plots/svm_loss_history.png', dpi=300)