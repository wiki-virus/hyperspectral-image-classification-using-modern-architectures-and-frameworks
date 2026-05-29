import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

NUM_BANDS = 204
NUM_STATS = 5

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

print("Loading saved backup Siamese model and extracting encoder...")
try:
    siamese_model = load_model("best_siamese_fewshot_backup.keras")
    encoder = None
    for layer in siamese_model.layers:
        if "Encoder" in layer.name or isinstance(layer, tf.keras.Model):
            encoder = layer
            break
    if encoder is not None:
        # Precompute embeddings
        print("Precomputing embeddings...")
        train_emb = encoder.predict(X_train, verbose=0)
        test_emb = encoder.predict(X_test, verbose=0)
        
        unique_classes = np.unique(y_train)
        class_indices = {c: np.where(y_train == c)[0] for c in unique_classes}
        
        for shots in [3, 5]:
            accs = []
            for trial in range(100):
                np.random.seed(trial)
                prototypes = {}
                for c in unique_classes:
                    chosen_idx = np.random.choice(class_indices[c], shots, replace=False)
                    prototypes[c] = np.mean(train_emb[chosen_idx], axis=0)
                
                # Evaluate on test set
                predictions = []
                for sample_emb in test_emb:
                    distances = {c: np.linalg.norm(sample_emb - prototypes[c]) for c in prototypes}
                    predicted_class = min(distances, key=distances.get)
                    predictions.append(predicted_class)
                accs.append(np.mean(np.array(predictions) == y_test) * 100)
                
            print(f"Few-shot accuracy of best Pure CNN encoder ({shots}-shot, 100 trials mean): {np.mean(accs):.2f}%")
            print(f"Few-shot accuracy of best Pure CNN encoder ({shots}-shot, max trial): {np.max(accs):.2f}%")
            print(f"Few-shot accuracy of best Pure CNN encoder ({shots}-shot, min trial): {np.min(accs):.2f}%")
    else:
        print("Encoder not found")
except Exception as e:
    print(f"Error: {e}")
