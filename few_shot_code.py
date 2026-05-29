# === CELL 0 ===
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure GPU acceleration and dynamic memory growth
print("Checking for GPU acceleration...")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] GPU Detected and Configured: {gpus}\n")
    except RuntimeError as e:
        print(f"[INFO] GPU is already initialized. Using existing configuration. Details: {e}\n")
else:
    print("[WARN] No GPU found. Falling back to CPU execution.\n")

# Global Configuration Parameters
NUM_BANDS = 204
NUM_STATS = 5
CLASS_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']


# === CELL 1 ===
# Load the raw hyperspectral data arrays
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

print(f"Loaded source tensor shape: {X_data.shape}")

# SPLIT 1: Take 80% for Training, leave 20% as "Temporary" (Stratified to keep distributions even)
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# SPLIT 2: Cut that 20% Temporary block in half for Validation and Local Testing
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Global Min-Max scaling calculated strictly from training bounds to prevent leakage
X_min = X_train_raw.min()
X_max = X_train_raw.max()

X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_val = (X_val_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

print(f"Training shapes:   {X_train.shape} (80%)")
print(f"Validation shapes: {X_val.shape} (10%)")
print(f"Local Test shapes: {X_test.shape} (10%)")


# === CELL 2 ===
def generate_siamese_pairs(X, y, pairs_to_generate=10000):
    """
    Creates structural pairs for similarity matching:
    - 50% Positive Pairs (Same cultivar class, target label = 1.0)
    - 50% Negative Pairs (Different cultivar classes, target label = 0.0)
    """
    pair_left = []
    pair_right = []
    labels = []

    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
    classes = list(class_indices.keys())

    for _ in range(pairs_to_generate):
        # 50% chance to generate a matching pair (Positive)
        if np.random.rand() > 0.5:
            current_class = np.random.choice(classes)
            idx1, idx2 = np.random.choice(class_indices[current_class], size=2, replace=False)
            pair_left.append(X[idx1])
            pair_right.append(X[idx2])
            labels.append(1.0)
        # 50% chance to generate a mismatching pair (Negative)
        else:
            class1, class2 = np.random.choice(classes, size=2, replace=False)
            idx1 = np.random.choice(class_indices[class1])
            idx2 = np.random.choice(class_indices[class2])
            pair_left.append(X[idx1])
            pair_right.append(X[idx2])
            labels.append(0.0)

    return [np.array(pair_left), np.array(pair_right)], np.array(labels)


def build_siamese_from_encoder(encoder, learning_rate=1e-4):
    """
    Constructs a Siamese neural network from a given encoder model.
    """
    input_left = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="left_leaf")
    input_right = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="right_leaf")

    encoded_left = encoder(input_left)
    encoded_right = encoder(input_right)

    l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([
        encoded_left,
        encoded_right
    ])
    prediction = layers.Dense(1, activation='sigmoid')(l1_distance)

    siamese = models.Model(inputs=[input_left, input_right], outputs=prediction)
    siamese.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    return siamese


def run_few_shot_inference(encoder, support_X, support_y, test_X, test_y, n_shots=5):
    """
    Executes a standard N-Shot Evaluation Pipeline:
    1. Creates class prototypes by averaging feature coordinates of limited samples.
    2. Maps hidden target profiles into encoder embeddings.
    3. Classifies targets using minimum geometric distance mapping.
    """
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


def print_few_shot_pass_fail(title, predicted_labels, true_labels):
    """Prints classification report and final holdout accuracy."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    correct_count = np.sum(predicted_labels == true_labels)
    total_samples = len(true_labels)
    final_accuracy = (correct_count / total_samples) * 100

    print(f"HOLDOUT ACCURACY: {correct_count}/{total_samples} ({final_accuracy:.2f}%)")
    print("-" * 60)
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))


def classify_crop_via_tournament(siamese_model, unknown_sample, reference_library, class_names, verbose=0):
    """
    Classifies a single unknown spectral signature by matching it against
    a reference library of known crop varieties using the Siamese network.
    """
    pair_left = []
    pair_right = []

    for name in class_names:
        known_reference = reference_library[name]
        pair_left.append(unknown_sample)
        pair_right.append(known_reference)

    pair_left = np.array(pair_left)
    pair_right = np.array(pair_right)

    similarity_scores = siamese_model.predict([pair_left, pair_right], verbose=verbose)
    scores = similarity_scores.flatten()

    best_match_idx = np.argmax(scores)
    predicted_crop = class_names[best_match_idx]

    return predicted_crop, scores


# === CELL 3 ===
# Generate balanced pairing matrices for training and validation runs
print("Generating Siamese matching pair arrays...")
X_train_pairs, y_train_pairs = generate_siamese_pairs(X_train, y_train, pairs_to_generate=10000)
X_val_pairs, y_val_pairs = generate_siamese_pairs(X_val, y_val, pairs_to_generate=2000)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
).shuffle(buffer_size=len(y_train_pairs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Siamese pair datasets configured.")


# === CELL 4 ===
def build_pure_cnn_encoder():
    """
    Constructs a pure feature extractor mapping structural 1D lines
    down onto a balanced geometric coordinate space.
    Configured for ~1M parameters and kernel sizes of 5.
    """
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))

    # Convolutional Feature Pipeline
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    # Projection down to a stable 32-dimensional geometric embedding space
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

print("Initializing Pure CNN Siamese Model (~1M params)...")
cnn_encoder = build_pure_cnn_encoder()
siamese_model = build_siamese_from_encoder(cnn_encoder, learning_rate=1e-4)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_fewshot.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)

# Verify parameter counts
cnn_encoder.summary()
siamese_model.summary()

print("Starting Siamese Pure CNN Few-Shot training (no early stopping)...")
with tf.device("/GPU:0"):
    history = siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=[checkpoint]
    )

# Save best core encoder
cnn_encoder.save("hyperleaf_fewshot_encoder.keras")
print("\n[SUCCESS] Pure CNN Encoder Model Saved Successfully.")


# === CELL 5 ===
def build_cnn_attention_encoder(embedding_dim=32):
    """CNN encoder with a lightweight self-attention block for few-shot embeddings (~1M params)."""
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))

    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    attention_output = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        dropout=0.05
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_FewShot_Encoder")

print("Initializing CNN + Attention Siamese Model (~1M params)...")
cnn_attention_encoder = build_cnn_attention_encoder(embedding_dim=32)
cnn_attention_siamese_model = build_siamese_from_encoder(cnn_attention_encoder, learning_rate=1e-4)

attention_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_attention_fewshot.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)

# Verify parameter counts
cnn_attention_encoder.summary()
cnn_attention_siamese_model.summary()

print("Starting CNN + Attention few-shot Siamese training (no early stopping)...")
with tf.device("/GPU:0"):
    cnn_attention_history = cnn_attention_siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        callbacks=[attention_checkpoint]
    )

cnn_attention_encoder.save("hyperleaf_fewshot_cnn_attention_encoder.keras")
print("\n[SUCCESS] CNN + Attention Encoder Model Saved Successfully.")


# === CELL 6 ===
# Clear previous Keras session and garbage collect to free up VRAM from prior runs
import gc
tf.keras.backend.clear_session()
gc.collect()

# Redefine datasets with a smaller batch size (32) for the Mamba training to prevent VRAM OOM
MAMBA_BATCH_SIZE = 32
print(f"Re-batching dataset to {MAMBA_BATCH_SIZE} for CNN + Mamba to prevent GPU OOM...")
mamba_train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
).shuffle(buffer_size=len(y_train_pairs)).batch(MAMBA_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

mamba_val_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
).batch(MAMBA_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class StableSSM(layers.Layer):
    """A compact, mathematically stable State-Space Model block compiled with XLA."""
    def __init__(self, d_model, d_state=16, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state

    def build(self, input_shape):
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_model, self.d_state),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.0, maxval=-1.0),
            trainable=True
        )
        self.B_proj = layers.Dense(self.d_state, use_bias=False)
        self.C_proj = layers.Dense(self.d_state, use_bias=False)
        self.dt_proj = layers.Dense(self.d_model, activation='softplus')
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        delta = self.dt_proj(x)
        B = self.B_proj(x)
        C = self.C_proj(x)

        A = -tf.exp(self.A_log)

        delta_expanded = tf.expand_dims(delta, -1)
        A_expanded = tf.expand_dims(A, 0)
        A_expanded = tf.expand_dims(A_expanded, 0)
        
        dA = delta_expanded * A_expanded
        A_bar = tf.exp(dA)

        B_expanded = tf.expand_dims(B, 2)
        x_expanded = tf.expand_dims(x, -1)
        Bx = x_expanded * B_expanded
        B_bar_x = delta_expanded * Bx

        A_bar_trans = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        B_bar_x_trans = tf.transpose(B_bar_x, perm=[1, 0, 2, 3])

        initial_state = tf.zeros((batch_size, self.d_model, self.d_state), dtype=tf.float32)

        def step(prev_h, inputs):
            a_bar_t, b_bar_x_t = inputs
            return a_bar_t * prev_h + b_bar_x_t

        h_seq = tf.scan(step, (A_bar_trans, B_bar_x_trans), initializer=initial_state)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])

        C_expanded = tf.expand_dims(C, 2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state
        })
        return config


def build_cnn_mamba_encoder(embedding_dim=32):
    """CNN encoder with a stable SSM block for few-shot embeddings (~1M params)."""
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))

    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    mamba_output = StableSSM(d_model=128, d_state=16)(x)
    x = layers.Add()([x, mamba_output])
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(96, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_FewShot_Encoder")

print("Initializing CNN + Mamba Siamese Model (~1M params)...")
cnn_mamba_encoder = build_cnn_mamba_encoder(embedding_dim=32)
cnn_mamba_siamese_model = build_siamese_from_encoder(cnn_mamba_encoder, learning_rate=1e-4)

mamba_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_mamba_fewshot.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)

# Verify parameter counts
cnn_mamba_encoder.summary()
cnn_mamba_siamese_model.summary()

print("Starting CNN + Mamba few-shot Siamese training (no early stopping)...")

with tf.device("/GPU:0"):
    cnn_mamba_history = cnn_mamba_siamese_model.fit(
        mamba_train_dataset,
        validation_data=mamba_val_dataset,
        epochs=200,
        callbacks=[mamba_checkpoint]
    )

cnn_mamba_encoder.save("hyperleaf_fewshot_cnn_mamba_encoder.keras")
print("\n[SUCCESS] CNN + Mamba Encoder Model Saved Successfully.")


# === CELL 7 ===
# ==========================================
# Siamese / Few-Shot Holdout Evaluation and Testing (5-Shot)
# ==========================================

import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading saved encoder and Siamese models from disk...")
cnn_encoder = load_model("hyperleaf_fewshot_encoder.keras")
cnn_attention_encoder = load_model("hyperleaf_fewshot_cnn_attention_encoder.keras")
cnn_mamba_encoder = load_model("hyperleaf_fewshot_cnn_mamba_encoder.keras", custom_objects={"StableSSM": StableSSM})
siamese_model = load_model("best_siamese_fewshot.keras")

print("Simulating 5-shot classification test on holdout samples for all encoders...")

# 1. Evaluate Pure CNN Encoder
predicted_labels_cnn = run_few_shot_inference(
    encoder=cnn_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=5
)
print_few_shot_pass_fail(
    title="Pure CNN Few-Shot 5-Shot Holdout Evaluation",
    predicted_labels=predicted_labels_cnn,
    true_labels=y_test
)

# 2. Evaluate CNN + Attention Encoder
predicted_labels_attention = run_few_shot_inference(
    encoder=cnn_attention_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=5
)
print_few_shot_pass_fail(
    title="CNN + Attention Few-Shot 5-Shot Holdout Evaluation",
    predicted_labels=predicted_labels_attention,
    true_labels=y_test
)

# 3. Evaluate CNN + Mamba Encoder
predicted_labels_mamba = run_few_shot_inference(
    encoder=cnn_mamba_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=5
)
print_few_shot_pass_fail(
    title="CNN + Mamba Few-Shot 5-Shot Holdout Evaluation",
    predicted_labels=predicted_labels_mamba,
    true_labels=y_test
)


# ==========================================
# Diagnostic Crop Classification Tournament (Using best model: Pure CNN Siamese)
# ==========================================
reference_library = {
    'Heerup': X_train[np.where(y_train == 0)[0][0]],
    'Kvium': X_train[np.where(y_train == 1)[0][0]],
    'Rembrandt': X_train[np.where(y_train == 2)[0][0]],
    'Sheriff': X_train[np.where(y_train == 3)[0][0]]
}

print("\nRunning Crop Classification Tournament on test set using the best Siamese model...")
count = 0
total_test = np.shape(X_test)[0]

for i in range(total_test):
    mystery_leaf = X_test[i]
    true_variety = CLASS_NAMES[y_test[i]]

    # Only print full logs for the first 5 samples to keep the notebook readable
    should_print = (i < 5)
    if should_print:
        print(f"\nSample #{i+1:03d} | Ground Truth Label: {true_variety}")

    predicted_variety, tournament_scores = classify_crop_via_tournament(
        siamese_model=siamese_model,
        unknown_sample=mystery_leaf,
        reference_library=reference_library,
        class_names=CLASS_NAMES,
        verbose=0
    )

    if should_print:
        for name, score in zip(CLASS_NAMES, tournament_scores):
            print(f"  Match Confidence with {name:10}: {score * 100:6.2f}%")
        print(f"  [WIN] Final Classifier Decision: {predicted_variety} (Correct: {predicted_variety == true_variety})")

    if predicted_variety == true_variety:
        count += 1

print("\n" + "=" * 60)
print(f"Tournament Holdout Accuracy: {count}/{total_test} ({count/total_test * 100:.2f}%)")
print("=" * 60)

