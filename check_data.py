import numpy as np
loaded_data = np.load("hyperleaf_processed.npz")
X = loaded_data['X']
y = loaded_data['y']
print("X shape:", X.shape)
print("y shape:", y.shape)
unique, counts = np.unique(y, return_counts=True)
print("Class counts:", dict(zip(unique, counts)))
print("Class names are:", ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff'])
