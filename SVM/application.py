import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1. DATASET DOWNLOAD
# -------------------------------------------------------------------
# Download the latest version of the MNIST dataset
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", path)

# -------------------------------------------------------------------
# 2. AUTOMATIC FILE SEARCH
# -------------------------------------------------------------------
files = []
for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        files.append(os.path.join(root, filename))

# Locate specific files based on MNIST naming conventions
X_train_path = next(f for f in files if 'train-images' in f)
y_train_path = next(f for f in files if 'train-labels' in f)
X_test_path = next(f for f in files if 't10k-images' in f)
y_test_path = next(f for f in files if 't10k-labels' in f)

# -------------------------------------------------------------------
# 3. SMART LOADING FUNCTIONS
# -------------------------------------------------------------------
def smart_load_images(filename):
    """ Tries to open as GZIP, falls back to standard binary if it fails """
    try:
        with gzip.open(filename, 'rb') as f:
            content = f.read()
    except Exception:
        with open(filename, 'rb') as f:
            content = f.read()

    # Offset 16 is standard for IDX image files
    data = np.frombuffer(content, np.uint8, offset=16)
    # Flatten the 28x28 images into 784-dimensional vectors
    return data.reshape(-1, 784)

def smart_load_labels(filename):
    """ Tries to open as GZIP, falls back to standard binary if it fails """
    try:
        with gzip.open(filename, 'rb') as f:
            content = f.read()
    except Exception:
        with open(filename, 'rb') as f:
            content = f.read()

    # Offset 8 is standard for IDX label files
    return np.frombuffer(content, np.uint8, offset=8)

# Load the full dataset
print("Loading files...")
X_train_full = smart_load_images(X_train_path)
y_train_full = smart_load_labels(y_train_path)
X_test_full = smart_load_images(X_test_path)
y_test_full = smart_load_labels(y_test_path)

# -------------------------------------------------------------------
# 4. VISUALIZATION FUNCTION
# -------------------------------------------------------------------
def visualize_samples(images, labels, n_samples=10):
    plt.figure(figsize=(15, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)

        # Reshape the 784 1D array back to 28x28 for image visualization
        image_2d = images[i].reshape(28, 28)

        plt.imshow(image_2d, cmap='gray')
        plt.title(f"Digit: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize before processing and training
print("Examples from the original dataset (28x28 pixels):")
visualize_samples(X_train_full, y_train_full, n_samples=10)

# -------------------------------------------------------------------
# 5. DATA PREPROCESSING
# -------------------------------------------------------------------
# Subsampling for training speed (SVM is computationally heavy: O(n^2) to O(n^3))
X_train = X_train_full[:10000]
y_train = y_train_full[:10000]
X_test = X_test_full[:2000]
y_test = y_test_full[:2000]

# Z-score Normalization
# SVMs are highly sensitive to unscaled data distances
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(np.float64))
X_test = scaler.transform(X_test.astype(np.float64))

# -------------------------------------------------------------------
# 6. MULTI-CLASS SVM TRAINING
# -------------------------------------------------------------------
# decision_function_shape='ovr' explicitly defines the One-vs-Rest strategy
# allowing the binary SVM to handle the 10 digits (0 to 9).
svm_model = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')

print(f"Training SVM with {X_train.shape[0]} samples...")
svm_model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 7. MODEL EVALUATION
# -------------------------------------------------------------------
y_pred = svm_model.predict(X_test)

print("\n" + "-" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("-" * 40)
print(classification_report(y_test, y_pred))