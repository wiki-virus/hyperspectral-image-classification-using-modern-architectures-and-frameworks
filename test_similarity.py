import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Load the dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

CLASS_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate_siamese_fast(siamese_model, encoder, support_X, support_y, test_X, test_y, n_shots=5):
    # Extract weights of the dense layer in Siamese
    dense_layer = None
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
            break
    
    if dense_layer is None:
        raise ValueError("Dense layer not found in Siamese model")
    
    w, b = dense_layer.get_weights() # w shape is (32, 1), b shape is (1,)
    w = w.flatten()
    b = b[0]
    
    # 1. Get embeddings for support and test
    unique_classes = np.unique(support_y)
    support_embeddings = {}
    for c in unique_classes:
        class_indices = np.where(support_y == c)[0]
        chosen_indices = class_indices[:n_shots]
        support_embeddings[c] = encoder.predict(support_X[chosen_indices], verbose=0) # shape (n_shots, 32)
        
    test_emb = encoder.predict(test_X, verbose=0) # shape (N_test, 32)
    
    # 2. Compute similarity for each test sample against the support samples
    predictions = []
    for i in range(len(test_X)):
        emb_i = test_emb[i] # shape (32,)
        class_scores = {}
        for c in unique_classes:
            supp_emb = support_embeddings[c] # shape (n_shots, 32)
            # L1 diff
            l1_diff = np.abs(emb_i - supp_emb) # shape (n_shots, 32)
            # Dense projection: w * l1_diff + b
            logits = np.dot(l1_diff, w) + b # shape (n_shots,)
            scores = sigmoid(logits)
            class_scores[c] = np.mean(scores)
        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)
        
    return np.mean(np.array(predictions) == test_y) * 100

print("Evaluating Pure CNN Siamese model...")
try:
    siamese_model = load_model("best_siamese_fewshot.keras")
    encoder = None
    for layer in siamese_model.layers:
        if "Encoder" in layer.name or isinstance(layer, tf.keras.Model):
            encoder = layer
            break
    if encoder is not None:
        acc = evaluate_siamese_fast(siamese_model, encoder, X_train, y_train, X_test, y_test, n_shots=5)
        print(f"Pure CNN Siamese Fast evaluation accuracy: {acc:.2f}%")
    else:
        print("Encoder not found")
except Exception as e:
    print(f"Error: {e}")
