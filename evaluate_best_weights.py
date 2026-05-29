import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

def run_few_shot_inference(encoder, support_X, support_y, test_X, test_y, n_shots=5):
    prototypes = {}
    for c in np.unique(support_y):
        class_indices = np.where(support_y == c)[0]
        chosen_indices = class_indices[:n_shots]
        chosen_samples = support_X[chosen_indices]
        embeddings = encoder.predict(chosen_samples, verbose=0)
        prototypes[c] = np.mean(embeddings, axis=0)

    test_embeddings = encoder.predict(test_X, verbose=0)
    predictions = []

    for sample_emb in test_embeddings:
        distances = {c: np.linalg.norm(sample_emb - prototypes[c]) for c in prototypes}
        predicted_class = min(distances, key=distances.get)
        predictions.append(predicted_class)

    return np.array(predictions)

print("--- Testing Siamese checkpoints ---")

# Try loading the Siamese model and extracting the encoder layer
try:
    siamese_model = load_model("best_siamese_fewshot.keras")
    # In Siamese model, the encoder is one of the layers. Let's find it.
    encoder = None
    for layer in siamese_model.layers:
        if "Encoder" in layer.name or isinstance(layer, tf.keras.Model):
            encoder = layer
            break
    if encoder is not None:
        print(f"Extracted encoder: {encoder.name}")
        preds = run_few_shot_inference(encoder, X_train, y_train, X_test, y_test, n_shots=5)
        acc = np.mean(preds == y_test) * 100
        print(f"Few-shot accuracy of best Pure CNN encoder: {acc:.2f}%")
    else:
        print("Could not find encoder layer in Siamese model.")
except Exception as e:
    print(f"Error evaluating pure CNN: {e}")

try:
    # Try the same for CNN + Attention
    attention_siamese = load_model("best_cnn_attention_fewshot.keras")
    encoder = None
    for layer in attention_siamese.layers:
        if "Encoder" in layer.name or isinstance(layer, tf.keras.Model):
            encoder = layer
            break
    if encoder is not None:
        print(f"Extracted encoder: {encoder.name}")
        preds = run_few_shot_inference(encoder, X_train, y_train, X_test, y_test, n_shots=5)
        acc = np.mean(preds == y_test) * 100
        print(f"Few-shot accuracy of best CNN + Attention encoder: {acc:.2f}%")
    else:
        print("Could not find encoder layer in Attention Siamese model.")
except Exception as e:
    print(f"Error evaluating CNN + Attention: {e}")
