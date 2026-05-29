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
        print(f"✅ GPU Detected and Configured: {gpus}\n")
    except RuntimeError as e:
        print(f"ℹ️ GPU is already initialized. Using existing configuration. Details: {e}\n")
else:
    print("⚠️ No GPU found. Falling back to CPU execution.\n")

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
np.shape(X_test)

# === CELL 3 ===
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

# Generate balanced pairing matrices for training and validation runs
print("Generating Siamese matching pair arrays...")
X_train_pairs, y_train_pairs = generate_siamese_pairs(X_train, y_train, pairs_to_generate=10000)
X_val_pairs, y_val_pairs = generate_siamese_pairs(X_val, y_val, pairs_to_generate=2000)
print("Pair arrays generated successfully.")

# === CELL 4 ===
"""HP_CONV_FILTERS = 64
HP_KERNEL_SIZE = 5
HP_EMBEDDING_DIM = 32
HP_LEARNING_RATE = 2e-4
HP_BATCH_SIZE = 256
HP_PAIR_COUNT = 10000"""

# === CELL 5 ===
def build_pure_cnn_encoder():
    """
    Constructs a pure feature extractor mapping structural 1D lines
    down onto a balanced geometric coordinate space.
    """
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))

    # Convolutional Feature Pipeline
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    # Projection down to a stable 32-dimensional geometric embedding space
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

# Instantiate the primary weight extractor
cnn_encoder = build_pure_cnn_encoder()

# Setup Dual Parallel Channels mapping through the exact same encoder block
input_left = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="left_leaf")
input_right = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="right_leaf")

encoded_left = cnn_encoder(input_left)
encoded_right = cnn_encoder(input_right)

# Compute the geometric absolute distance matrix between signatures
l1_distance_layer = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
l1_distance = l1_distance_layer([encoded_left, encoded_right])

# Binary Output Classification Head (1.0 = Same variety, 0.0 = Mismatch)
prediction = layers.Dense(1, activation='sigmoid')(l1_distance)

siamese_model = models.Model(inputs=[input_left, input_right], outputs=prediction)
siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

siamese_model.summary()

# === CELL 6 ===
# Optimize training pipelines for hardware acceleration with high batch size
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
).shuffle(buffer_size=len(y_train_pairs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model optimization metrics tracking checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_fewshot.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
)

print("Starting Siamese Few-Shot verification training on GPU...")
with tf.device("/GPU:0"):
    history = siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[checkpoint, early_stop]
    )

# Extract and save the trained standalone core encoder net
cnn_encoder.save("hyperleaf_fewshot_encoder.keras")
print("\n🎉 Core Encoder Model Saved Successfully.")

# === CELL 7 ===
def run_few_shot_inference(encoder, support_X, support_y, test_X, test_y, n_shots=3):
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
    """Prints sample-by-sample PASS/FAIL plus summary metrics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    correct_count = 0
    total_samples = len(true_labels)

    for i, (pred_idx, true_idx) in enumerate(zip(predicted_labels, true_labels), start=1):
        pred_name = CLASS_NAMES[int(pred_idx)]
        true_name = CLASS_NAMES[int(true_idx)]

        if pred_idx == true_idx:
            status = "PASS"
            correct_count += 1
        else:
            status = "FAIL"

        print(f"Sample #{i:03d} | Predicted: {pred_name: <10} | Actual: {true_name: <10} | {status}")

    final_accuracy = (correct_count / total_samples) * 100
    print("-" * 60)
    print(f"HOLDOUT ACCURACY: {correct_count}/{total_samples} ({final_accuracy:.2f}%)")
    print("-" * 60)

    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))


print("Simulating a 3-shot classification test on holdout samples...")
predicted_labels = run_few_shot_inference(
    encoder=cnn_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=3
)

print_few_shot_pass_fail(
    title="Pure CNN Few-Shot 3-Shot Holdout Evaluation",
    predicted_labels=predicted_labels,
    true_labels=y_test
)

# === CELL 9 ===
def build_cnn_attention_encoder(embedding_dim=32):
    """CNN encoder with a lightweight self-attention block for few-shot embeddings."""
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
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    normalized_embeddings = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1)
    )(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_FewShot_Encoder")


def build_siamese_from_encoder(encoder, learning_rate=1e-4):
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
        metrics=['accuracy']
    )
    return siamese


cnn_attention_encoder = build_cnn_attention_encoder(embedding_dim=32)
cnn_attention_siamese_model = build_siamese_from_encoder(cnn_attention_encoder, learning_rate=1e-4)
cnn_attention_siamese_model.summary()

attention_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_attention_fewshot.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
attention_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

print("Starting CNN + Attention few-shot Siamese training...")
with tf.device("/GPU:0"):
    cnn_attention_history = cnn_attention_siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[attention_checkpoint, attention_early_stop]
    )

cnn_attention_encoder.save("hyperleaf_fewshot_cnn_attention_encoder.keras")
print("CNN + Attention few-shot encoder saved.")

print("\nCNN + Attention 3-shot holdout evaluation:")
cnn_attention_predicted_labels = run_few_shot_inference(
    encoder=cnn_attention_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=3
)
print_few_shot_pass_fail(
    title="CNN + Attention Few-Shot 3-Shot Holdout Evaluation",
    predicted_labels=cnn_attention_predicted_labels,
    true_labels=y_test
)


# === CELL 11 ===
class MambaBlock(layers.Layer):
    """A compact Mamba-inspired state-space block for spectral sequence embeddings."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

    def build(self, input_shape):
        self.in_proj = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d = layers.Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            padding='same',
            groups=self.d_inner,
            activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)

        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)

        delta_x = delta * x_branch
        state_influence = tf.reduce_mean(B, axis=-1, keepdims=True)
        decay_factors = tf.exp(-delta * tf.abs(state_influence))
        cum_decay = tf.math.cumprod(decay_factors, axis=1)
        outputs = tf.math.cumsum(delta_x * cum_decay, axis=1) / (cum_decay + 1e-8)

        outputs = outputs * tf.reduce_mean(C, axis=-1, keepdims=True)
        gated_output = outputs * tf.keras.activations.swish(res_branch)
        return self.out_proj(gated_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand
        })
        return config


def build_cnn_mamba_encoder(embedding_dim=32):
    """CNN encoder with a Mamba-inspired sequence block for few-shot embeddings."""
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))

    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)

    mamba_output = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_output])
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    normalized_embeddings = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1)
    )(raw_embeddings)

    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_FewShot_Encoder")


cnn_mamba_encoder = build_cnn_mamba_encoder(embedding_dim=32)
cnn_mamba_siamese_model = build_siamese_from_encoder(cnn_mamba_encoder, learning_rate=1e-4)
cnn_mamba_siamese_model.summary()

mamba_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_mamba_fewshot.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
mamba_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

print("Starting CNN + Mamba few-shot Siamese training...")
with tf.device("/GPU:0"):
    cnn_mamba_history = cnn_mamba_siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[mamba_checkpoint, mamba_early_stop]
    )

cnn_mamba_encoder.save("hyperleaf_fewshot_cnn_mamba_encoder.keras")
print("CNN + Mamba few-shot encoder saved.")

print("\nCNN + Mamba 3-shot holdout evaluation:")
cnn_mamba_predicted_labels = run_few_shot_inference(
    encoder=cnn_mamba_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=3
)
print_few_shot_pass_fail(
    title="CNN + Mamba Few-Shot 3-Shot Holdout Evaluation",
    predicted_labels=cnn_mamba_predicted_labels,
    true_labels=y_test
)


# === CELL 12 ===
# ==========================================
# 7. Single-Sample Multiclass Crop Classification Tournament
# ==========================================

def classify_crop_via_tournament(siamese_model, unknown_sample, reference_library, class_names):
    """
    Classifies a single unknown spectral signature by matching it against
    a reference library of known crop varieties using the Siamese network.
    """
    pair_left = []
    pair_right = []

    # 1. Create a verification pair for every known crop category
    for name in class_names:
        known_reference = reference_library[name]

        pair_left.append(unknown_sample)  # Head 1: The completely unknown crop
        pair_right.append(known_reference)  # Head 2: The known baseline reference variety

    # Convert lists to model-friendly numpy tensors
    pair_left = np.array(pair_left)
    pair_right = np.array(pair_right)

    # 2. Run all 4 pairs through the network at the exact same time
    similarity_scores = siamese_model.predict([pair_left, pair_right], verbose=1)
    scores = similarity_scores.flatten()

    # 3. Print out the diagnostic scoreboard
    print("\n--- Siamese Tournament Brackets ---")
    for name, score in zip(class_names, scores):
        print(f"Match Confidence with {name:10}: {score * 100:6.2f}%")

    # 4. Find the highest matching index (the score closest to 1.0)
    best_match_idx = np.argmax(scores)
    predicted_crop = class_names[best_match_idx]

    return predicted_crop, scores


# --- EXECUTE DIAGNOSTIC TEST RUN ---

# 1. Build a reference library using the first available sample of each variety from X_train
reference_library = {
    'Heerup': X_train[np.where(y_train == 0)[0][0]],
    'Kvium': X_train[np.where(y_train == 1)[0][0]],
    'Rembrandt': X_train[np.where(y_train == 2)[0][0]],
    'Sheriff': X_train[np.where(y_train == 3)[0][0]]
}
count=0
for i in range(1,np.shape(X_test)[0]):
    test_index = i
    mystery_leaf = X_test[test_index]
    true_variety = CLASS_NAMES[y_test[test_index]]

    print(f"Ground Truth Label (What it actually is): {true_variety}")

    predicted_variety, tournament_scores = classify_crop_via_tournament(
        siamese_model=siamese_model,
        unknown_sample=mystery_leaf,
        reference_library=reference_library,
        class_names=CLASS_NAMES
    )


    if(predicted_variety == true_variety):
        print("✅")
        count+=1
    else:
        print('❌')
        print(f"\n🏆 Final Classifier Decision: This spectral signature belongs to a : {predicted_variety} crop!")
print(count, count/158)

# === CELL 13 ===

