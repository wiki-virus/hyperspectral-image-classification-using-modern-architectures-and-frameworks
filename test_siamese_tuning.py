import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

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
X_val = (X_val_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

def generate_siamese_pairs(X, y, pairs_to_generate=10000):
    pair_left, pair_right, labels = [], [], []
    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
    classes = list(class_indices.keys())
    for _ in range(pairs_to_generate):
        if np.random.rand() > 0.5:
            current_class = np.random.choice(classes)
            idx1, idx2 = np.random.choice(class_indices[current_class], size=2, replace=False)
            pair_left.append(X[idx1])
            pair_right.append(X[idx2])
            labels.append(1.0)
        else:
            class1, class2 = np.random.choice(classes, size=2, replace=False)
            idx1 = np.random.choice(class_indices[class1])
            idx2 = np.random.choice(class_indices[class2])
            pair_left.append(X[idx1])
            pair_right.append(X[idx2])
            labels.append(0.0)
    return [np.array(pair_left), np.array(pair_right)], np.array(labels)

def evaluate_encoder(encoder):
    test_embeddings = encoder.predict(X_test, verbose=0)
    train_embeddings = encoder.predict(X_train, verbose=0)
    
    unique_classes = np.unique(y_train)
    train_class_indices = {c: np.where(y_train == c)[0] for c in unique_classes}
    
    accs = []
    for trial in range(100):
        np.random.seed(trial)
        prototypes = {}
        for c in unique_classes:
            chosen_idx = np.random.choice(train_class_indices[c], 5, replace=False)
            prototypes[c] = np.mean(train_embeddings[chosen_idx], axis=0)
        
        preds = []
        for sample_emb in test_embeddings:
            dists = {c: np.linalg.norm(sample_emb - prototypes[c]) for c in prototypes}
            preds.append(min(dists, key=dists.get))
        
        acc = np.mean(np.array(preds) == y_test) * 100
        accs.append(acc)
    return np.mean(accs)

# Let's try training with different learning rates and L2 regularization
for lr in [2e-4, 5e-4]:
    for l2_reg in [1e-5, 1e-4]:
        print(f"\nTesting LR: {lr}, L2 Reg: {l2_reg}")
        
        inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
        reg = tf.keras.regularizers.l2(l2_reg)
        
        x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
        raw_embeddings = layers.Dense(32, activation=None)(x)
        normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
        
        encoder = models.Model(inputs=inputs, outputs=normalized_embeddings)
        
        # Build Siamese
        input_left = layers.Input(shape=(NUM_BANDS, NUM_STATS))
        input_right = layers.Input(shape=(NUM_BANDS, NUM_STATS))
        encoded_left = encoder(input_left)
        encoded_right = encoder(input_right)
        l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_left, encoded_right])
        prediction = layers.Dense(1, activation='sigmoid')(l1_distance)
        siamese = models.Model(inputs=[input_left, input_right], outputs=prediction)
        
        siamese.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        X_train_pairs, y_train_pairs = generate_siamese_pairs(X_train, y_train, pairs_to_generate=15000)
        X_val_pairs, y_val_pairs = generate_siamese_pairs(X_val, y_val, pairs_to_generate=3000)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
        ).shuffle(buffer_size=len(y_train_pairs)).batch(256).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
        ).batch(256).prefetch(tf.data.AUTOTUNE)
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
        )
        
        siamese.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=[early_stop],
            verbose=0
        )
        
        mean_acc = evaluate_encoder(encoder)
        print(f"==> Mean 5-Shot Accuracy: {mean_acc:.2f}%")
