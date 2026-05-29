import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Created {filename}")

# ==================== SIAMESE NOTEBOOK CELLS ====================
siamese_cells = [
    ("markdown", "# Few-Shot Learning using Siamese Networks\nThis notebook trains and evaluates four architectures (CNN, CNN+Attention, CNN+Mamba, CNN+Attention+Mamba) on the HyperLeaf spectral dataset using Siamese Networks."),
    ("code", """import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU configuration error:", e)

# Parameters
NUM_BANDS = 204
NUM_STATS = 5

# Load dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

# Stratified train/val/test split
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_val = (X_val_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

# Function to generate Siamese pairs
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

X_train_pairs, y_train_pairs = generate_siamese_pairs(X_train, y_train, pairs_to_generate=15000)
X_val_pairs, y_val_pairs = generate_siamese_pairs(X_val, y_val, pairs_to_generate=3000)

BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
).shuffle(buffer_size=len(y_train_pairs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Helper function to compile Siamese model
def build_siamese_model(encoder, learning_rate=2e-4):
    input_left = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    input_right = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    encoded_left = encoder(input_left)
    encoded_right = encoder(input_right)
    l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_left, encoded_right])
    prediction = layers.Dense(1, activation='sigmoid')(l1_distance)
    siamese = models.Model(inputs=[input_left, input_right], outputs=prediction)
    siamese.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return siamese"""),
    ("code", """# --- 1. CNN Architecture ---
def build_cnn_encoder(l2_reg=1e-4):
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
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

cnn_encoder = build_cnn_encoder()
siamese_cnn = build_siamese_model(cnn_encoder)

checkpoint_cnn = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_cnn.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

print("Training CNN Siamese...")
history_cnn = siamese_cnn.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_cnn, early_stop],
    verbose=1
)"""),
    ("code", """# --- 2. CNN + Attention Architecture ---
def build_cnn_attention_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Attention Block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Encoder")

cnn_attention_encoder = build_cnn_attention_encoder()
siamese_cnn_attention = build_siamese_model(cnn_attention_encoder)

checkpoint_attention = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_cnn_attention.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

print("Training CNN+Attention Siamese...")
history_attention = siamese_cnn_attention.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_attention, early_stop],
    verbose=1
)"""),
    ("code", """# --- 3. CNN + Mamba Architecture ---
class StableMambaBlock(layers.Layer):
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
            filters=self.d_inner, kernel_size=self.d_conv, padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        A_init = tf.keras.initializers.Constant(A_init_val)
        self.A = self.add_weight(
            name="A",
            shape=(self.d_inner, self.d_state),
            initializer=A_init,
            trainable=True
        )
        
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)
        
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(self.A, axis=0)
        A_expanded = tf.expand_dims(A_expanded, axis=0)
        
        A_bar = tf.exp(delta_expanded * A_expanded)
        
        B_expanded = tf.expand_dims(B, axis=2)
        B_bar = delta_expanded * B_expanded
        
        x_branch_expanded = tf.expand_dims(x_branch, axis=-1)
        inputs_ssm = B_bar * x_branch_expanded
        
        inputs_ssm = tf.transpose(inputs_ssm, perm=[1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        
        elems = (A_bar, inputs_ssm)
        batch_size = tf.shape(x)[0]
        h_0 = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=tf.float32)
        
        def scan_fn(h_prev, elem):
            a_bar_t, ssm_in_t = elem
            return a_bar_t * h_prev + ssm_in_t
        
        h_seq = tf.scan(scan_fn, elems, initializer=h_0)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])
        
        C_expanded = tf.expand_dims(C, axis=2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        
        D_expanded = tf.reshape(self.D, (1, 1, self.d_inner))
        y = y + D_expanded * x_branch
        
        gated_output = y * tf.keras.activations.swish(res_branch)
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

def build_cnn_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    # Mamba block
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_Encoder")

cnn_mamba_encoder = build_cnn_mamba_encoder()
siamese_cnn_mamba = build_siamese_model(cnn_mamba_encoder)

checkpoint_mamba = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_cnn_mamba.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

print("Training CNN+Mamba Siamese...")
history_mamba = siamese_cnn_mamba.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_mamba, early_stop],
    verbose=1
)"""),
    ("code", """# --- 4. CNN + Attention + Mamba Architecture ---
def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    # Attention block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    # Mamba block
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Mamba_Encoder")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
siamese_cnn_attention_mamba = build_siamese_model(cnn_attention_mamba_encoder)

checkpoint_attn_mamba = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_cnn_attention_mamba.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

print("Training CNN+Attention+Mamba Siamese...")
history_attn_mamba = siamese_cnn_attention_mamba.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_attn_mamba, early_stop],
    verbose=1
)"""),
    ("code", """# --- 5. Testing Cell (Holdout Evaluation) ---
# 5-Shot Holdout Evaluation over 100 trials for each model
unique_classes = np.unique(y_train)
train_class_indices = {c: np.where(y_train == c)[0] for c in unique_classes}

encoders = {
    'CNN': cnn_encoder,
    'CNN+Attention': cnn_attention_encoder,
    'CNN+Mamba': cnn_mamba_encoder,
    'CNN+Attention+Mamba': cnn_attention_mamba_encoder
}

eval_results = {}

for name, encoder in encoders.items():
    print(f"\\nEvaluating {name} Encoder...")
    suffix = name.lower().replace('+', '_')
    try:
        encoder.load_weights(f"best_siamese_{suffix}.keras", by_name=True, skip_mismatch=True)
    except Exception as e:
        print(f"No saved weights found for {name}, using current weights.")
        
    test_embeddings = encoder.predict(X_test, verbose=0)
    train_embeddings = encoder.predict(X_train, verbose=0)
    
    trial_accs = []
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
        trial_accs.append(acc)
    
    mean_acc = np.mean(trial_accs)
    max_acc = np.max(trial_accs)
    min_acc = np.min(trial_accs)
    eval_results[name] = trial_accs
    
    print(f"5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")"""),
    ("code", """# --- 6. Plotting Cell ---
# Plotting training curves and evaluation results
histories = {
    'CNN': history_cnn.history if 'history_cnn' in locals() and hasattr(history_cnn, 'history') else None,
    'CNN+Attention': history_attention.history if 'history_attention' in locals() and hasattr(history_attention, 'history') else None,
    'CNN+Mamba': history_mamba.history if 'history_mamba' in locals() and hasattr(history_mamba, 'history') else None,
    'CNN+Attention+Mamba': history_attn_mamba.history if 'history_attn_mamba' in locals() and hasattr(history_attn_mamba, 'history') else None
}

# 1. Loss vs Epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    if hist:
        plt.plot(hist['loss'], label=f'{name} Train')
        plt.plot(hist['val_loss'], linestyle='--', label=f'{name} Val')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy vs Epochs
plt.subplot(1, 2, 2)
for name, hist in histories.items():
    if hist:
        plt.plot(hist['accuracy'], label=f'{name} Train')
        plt.plot(hist['val_accuracy'], linestyle='--', label=f'{name} Val')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Holdout Evaluation Distributions
if eval_results:
    plt.figure(figsize=(10, 6))
    data_to_plot = [eval_results[name] for name in eval_results]
    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Test Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()""")
]

# ==================== PROTONET NOTEBOOK CELLS ====================
protonet_cells = [
    ("markdown", "# Few-Shot Learning using Prototypical Networks (ProtoNet)\nThis notebook trains and evaluates four architectures (CNN, CNN+Attention, CNN+Mamba, CNN+Attention+Mamba) on the HyperLeaf spectral dataset using Prototypical Networks."),
    ("code", """import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU configuration error:", e)

# Parameters
NUM_BANDS = 204
NUM_STATS = 5

# Load dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

# Stratified train/val/test split
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

# Function to sample episodic few-shot batches
def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=5):
    unique_classes = np.unique(y)
    selected_classes = unique_classes[:n_way]
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        sampled_indices = np.random.choice(class_indices, k_shot + q_query, replace=False)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.concatenate(query_y, axis=0)
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))

# Prototypical Network model class
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    def call(self, support_x, support_y, query_x, n_way, k_shot):
        support_embeddings = self.encoder(support_x, training=True)
        query_embeddings = self.encoder(query_x, training=True)
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1)
        query_expanded = tf.expand_dims(query_embeddings, axis=1)
        proto_expanded = tf.expand_dims(prototypes, axis=0)
        distances = tf.reduce_sum(tf.square(query_expanded - proto_expanded), axis=-1)
        return -distances

def train_step_protonet(model, optimizer, support_x, support_y, query_x, query_y, n_way, k_shot):
    with tf.GradientTape() as tape:
        logits = model(support_x, support_y, query_x, n_way, k_shot)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, logits, from_logits=True
        ))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, accuracy

def train_protonet_model(encoder, episodes=1000, lr=2e-4):
    proto_net = PrototypicalNetwork(encoder)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_history, acc_history = [], []
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_protonet(proto_net, optimizer, sx, sy, qx, qy, 4, 5)
        loss_history.append(loss.numpy())
        acc_history.append(acc.numpy())
        if ep % 200 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    return loss_history, acc_history"""),
    ("code", """# --- 1. CNN Architecture ---
def build_cnn_encoder(l2_reg=1e-4):
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
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

cnn_encoder = build_cnn_encoder()
print("Training CNN ProtoNet...")
loss_cnn, acc_cnn = train_protonet_model(cnn_encoder, episodes=1000)
cnn_encoder.save("best_protonet_cnn.keras")"""),
    ("code", """# --- 2. CNN + Attention Architecture ---
def build_cnn_attention_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Attention block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Encoder")

cnn_attention_encoder = build_cnn_attention_encoder()
print("Training CNN+Attention ProtoNet...")
loss_attention, acc_attention = train_protonet_model(cnn_attention_encoder, episodes=1000)
cnn_attention_encoder.save("best_protonet_cnn_attention.keras")"""),
    ("code", """# --- 3. CNN + Mamba Architecture ---
class StableMambaBlock(layers.Layer):
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
            filters=self.d_inner, kernel_size=self.d_conv, padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        A_init = tf.keras.initializers.Constant(A_init_val)
        self.A = self.add_weight(
            name="A",
            shape=(self.d_inner, self.d_state),
            initializer=A_init,
            trainable=True
        )
        
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)
        
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(self.A, axis=0)
        A_expanded = tf.expand_dims(A_expanded, axis=0)
        
        A_bar = tf.exp(delta_expanded * A_expanded)
        
        B_expanded = tf.expand_dims(B, axis=2)
        B_bar = delta_expanded * B_expanded
        
        x_branch_expanded = tf.expand_dims(x_branch, axis=-1)
        inputs_ssm = B_bar * x_branch_expanded
        
        inputs_ssm = tf.transpose(inputs_ssm, perm=[1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        
        elems = (A_bar, inputs_ssm)
        batch_size = tf.shape(x)[0]
        h_0 = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=tf.float32)
        
        def scan_fn(h_prev, elem):
            a_bar_t, ssm_in_t = elem
            return a_bar_t * h_prev + ssm_in_t
        
        h_seq = tf.scan(scan_fn, elems, initializer=h_0)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])
        
        C_expanded = tf.expand_dims(C, axis=2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        
        D_expanded = tf.reshape(self.D, (1, 1, self.d_inner))
        y = y + D_expanded * x_branch
        
        gated_output = y * tf.keras.activations.swish(res_branch)
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

def build_cnn_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_Encoder")

cnn_mamba_encoder = build_cnn_mamba_encoder()
print("Training CNN+Mamba ProtoNet...")
loss_mamba, acc_mamba = train_protonet_model(cnn_mamba_encoder, episodes=1000)
cnn_mamba_encoder.save("best_protonet_cnn_mamba.keras")"""),
    ("code", """# --- 4. CNN + Attention + Mamba Architecture ---
def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Mamba_Encoder")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
print("Training CNN+Attention+Mamba ProtoNet...")
loss_attn_mamba, acc_attn_mamba = train_protonet_model(cnn_attention_mamba_encoder, episodes=1000)
cnn_attention_mamba_encoder.save("best_protonet_cnn_attention_mamba.keras")"""),
    ("code", """# --- 5. Testing Cell (Holdout Evaluation) ---
# 5-Shot Holdout Evaluation over 100 trials for each model
encoders = {
    'CNN': cnn_encoder,
    'CNN+Attention': cnn_attention_encoder,
    'CNN+Mamba': cnn_mamba_encoder,
    'CNN+Attention+Mamba': cnn_attention_mamba_encoder
}

eval_results = {}

for name, encoder in encoders.items():
    print(f"\\nEvaluating {name} ProtoNet Encoder...")
    suffix = name.lower().replace('+', '_')
    try:
        encoder.load_weights(f"best_protonet_{suffix}.keras", by_name=True, skip_mismatch=True)
    except Exception as e:
        print(f"No saved weights found for {name}, using current weights.")
        
    trial_accs = []
    for i in range(100):
        # Sample test episode (5-shot, 5-query)
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        
        # Extract embeddings
        support_embeddings = encoder(sx, training=False)
        query_embeddings = encoder(qx, training=False)
        
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (4, 5, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1)
        
        query_expanded = tf.expand_dims(query_embeddings, axis=1)
        proto_expanded = tf.expand_dims(prototypes, axis=0)
        
        distances = tf.reduce_sum(tf.square(query_expanded - proto_expanded), axis=-1)
        preds = tf.argmax(-distances, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)
        
    mean_acc = np.mean(trial_accs)
    max_acc = np.max(trial_accs)
    min_acc = np.min(trial_accs)
    eval_results[name] = trial_accs
    
    print(f"5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")"""),
    ("code", """# --- 6. Plotting Cell ---
# Plotting training curves and evaluation results
histories = {
    'CNN': (loss_cnn, acc_cnn) if 'loss_cnn' in locals() else None,
    'CNN+Attention': (loss_attention, acc_attention) if 'loss_attention' in locals() else None,
    'CNN+Mamba': (loss_mamba, acc_mamba) if 'loss_mamba' in locals() else None,
    'CNN+Attention+Mamba': (loss_attn_mamba, acc_attn_mamba) if 'loss_attn_mamba' in locals() else None
}

# 1. Loss vs Episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[0], label=f'{name}')
plt.title('Loss vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy vs Episode
plt.subplot(1, 2, 2)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[1], label=f'{name}')
plt.title('Accuracy vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Holdout Evaluation Distributions
if eval_results:
    plt.figure(figsize=(10, 6))
    data_to_plot = [eval_results[name] for name in eval_results]
    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Test Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()""")
]

# ==================== RELATIONNET NOTEBOOK CELLS ====================
relationnet_cells = [
    ("markdown", "# Few-Shot Learning using Relation Networks (RelationNet)\nThis notebook trains and evaluates four architectures (CNN, CNN+Attention, CNN+Mamba, CNN+Attention+Mamba) on the HyperLeaf spectral dataset using Relation Networks."),
    ("code", """import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU configuration error:", e)

# Parameters
NUM_BANDS = 204
NUM_STATS = 5

# Load dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

# Stratified train/val/test split
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

# Function to sample episodic few-shot batches
def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=5):
    unique_classes = np.unique(y)
    selected_classes = unique_classes[:n_way]
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        sampled_indices = np.random.choice(class_indices, k_shot + q_query, replace=False)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.concatenate(query_y, axis=0)
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))

# Relation module definition
def build_relation_module(embedding_dim=32, l2_reg=1e-4):
    inputs = layers.Input(shape=(4 * embedding_dim,))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(64, activation=None, kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(32, activation=None, kernel_regularizer=reg)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    outputs = layers.Dense(1, activation=None)(x)
    return models.Model(inputs=inputs, outputs=outputs, name="Relation_Module")

class RelationNetwork(tf.keras.Model):
    def __init__(self, encoder, relation_module):
        super(RelationNetwork, self).__init__()
        self.encoder = encoder
        self.relation_module = relation_module
        
    def call(self, support_x, support_y, query_x, n_way, k_shot, training=True):
        support_embeddings = self.encoder(support_x, training=training)
        query_embeddings = self.encoder(query_x, training=training)
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1)
        num_queries = tf.shape(query_embeddings)[0]
        
        # Compare each query with prototypes
        query_expanded = tf.expand_dims(query_embeddings, axis=1)
        query_tiled = tf.tile(query_expanded, [1, n_way, 1])
        proto_expanded = tf.expand_dims(prototypes, axis=0)
        proto_tiled = tf.tile(proto_expanded, [num_queries, 1, 1])
        
        diff = tf.abs(query_tiled - proto_tiled)
        prod = query_tiled * proto_tiled
        relation_input = tf.concat([query_tiled, proto_tiled, diff, prod], axis=-1)
        
        flat_shape = tf.shape(relation_input)
        flat_input = tf.reshape(relation_input, (flat_shape[0] * flat_shape[1], flat_shape[2]))
        relation_logits = self.relation_module(flat_input, training=training)
        relation_logits = tf.reshape(relation_logits, (flat_shape[0], flat_shape[1]))
        return relation_logits

def train_step_relationnet(model, optimizer, support_x, support_y, query_x, query_y, n_way, k_shot):
    with tf.GradientTape() as tape:
        logits = model(support_x, support_y, query_x, n_way, k_shot, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, logits, from_logits=True
        ))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, accuracy

def train_relation_model(encoder, relation_module, episodes=1000, lr=2e-4):
    relation_net = RelationNetwork(encoder, relation_module)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_history, acc_history = [], []
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_relationnet(relation_net, optimizer, sx, sy, qx, qy, 4, 5)
        loss_history.append(loss.numpy())
        acc_history.append(acc.numpy())
        if ep % 200 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    return loss_history, acc_history"""),
    ("code", """# --- 1. CNN Architecture ---
def build_cnn_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

cnn_encoder = build_cnn_encoder()
cnn_rel_module = build_relation_module()
print("Training CNN RelationNet...")
loss_cnn, acc_cnn = train_relation_model(cnn_encoder, cnn_rel_module, episodes=1000)
cnn_encoder.save("best_relationnet_cnn_encoder.keras")
cnn_rel_module.save("best_relationnet_cnn_module.keras")"""),
    ("code", """# --- 2. CNN + Attention Architecture ---
def build_cnn_attention_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Attention block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Encoder")

cnn_attention_encoder = build_cnn_attention_encoder()
cnn_attention_rel_module = build_relation_module()
print("Training CNN+Attention RelationNet...")
loss_attention, acc_attention = train_relation_model(cnn_attention_encoder, cnn_attention_rel_module, episodes=1000)
cnn_attention_encoder.save("best_relationnet_cnn_attention_encoder.keras")
cnn_attention_rel_module.save("best_relationnet_cnn_attention_module.keras")"""),
    ("code", """# --- 3. CNN + Mamba Architecture ---
class StableMambaBlock(layers.Layer):
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
            filters=self.d_inner, kernel_size=self.d_conv, padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        A_init = tf.keras.initializers.Constant(A_init_val)
        self.A = self.add_weight(
            name="A",
            shape=(self.d_inner, self.d_state),
            initializer=A_init,
            trainable=True
        )
        
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)
        
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(self.A, axis=0)
        A_expanded = tf.expand_dims(A_expanded, axis=0)
        
        A_bar = tf.exp(delta_expanded * A_expanded)
        
        B_expanded = tf.expand_dims(B, axis=2)
        B_bar = delta_expanded * B_expanded
        
        x_branch_expanded = tf.expand_dims(x_branch, axis=-1)
        inputs_ssm = B_bar * x_branch_expanded
        
        inputs_ssm = tf.transpose(inputs_ssm, perm=[1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        
        elems = (A_bar, inputs_ssm)
        batch_size = tf.shape(x)[0]
        h_0 = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=tf.float32)
        
        def scan_fn(h_prev, elem):
            a_bar_t, ssm_in_t = elem
            return a_bar_t * h_prev + ssm_in_t
        
        h_seq = tf.scan(scan_fn, elems, initializer=h_0)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])
        
        C_expanded = tf.expand_dims(C, axis=2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        
        D_expanded = tf.reshape(self.D, (1, 1, self.d_inner))
        y = y + D_expanded * x_branch
        
        gated_output = y * tf.keras.activations.swish(res_branch)
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

def build_cnn_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_Encoder")

cnn_mamba_encoder = build_cnn_mamba_encoder()
cnn_mamba_rel_module = build_relation_module()
print("Training CNN+Mamba RelationNet...")
loss_mamba, acc_mamba = train_relation_model(cnn_mamba_encoder, cnn_mamba_rel_module, episodes=1000)
cnn_mamba_encoder.save("best_relationnet_cnn_mamba_encoder.keras")
cnn_mamba_rel_module.save("best_relationnet_cnn_mamba_module.keras")"""),
    ("code", """# --- 4. CNN + Attention + Mamba Architecture ---
def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Mamba_Encoder")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
cnn_attention_mamba_rel_module = build_relation_module()
print("Training CNN+Attention+Mamba RelationNet...")
loss_attn_mamba, acc_attn_mamba = train_relation_model(cnn_attention_mamba_encoder, cnn_attention_mamba_rel_module, episodes=1000)
cnn_attention_mamba_encoder.save("best_relationnet_cnn_attention_mamba_encoder.keras")
cnn_attention_mamba_rel_module.save("best_relationnet_cnn_attention_mamba_module.keras")"""),
    ("code", """# --- 5. Testing Cell (Holdout Evaluation) ---
# 5-Shot Holdout Evaluation over 100 trials for each model
configurations = {
    'CNN': (cnn_encoder, cnn_rel_module),
    'CNN+Attention': (cnn_attention_encoder, cnn_attention_rel_module),
    'CNN+Mamba': (cnn_mamba_encoder, cnn_mamba_rel_module),
    'CNN+Attention+Mamba': (cnn_attention_mamba_encoder, cnn_attention_mamba_rel_module)
}

eval_results = {}

for name, (encoder, rel_module) in configurations.items():
    print(f"\\nEvaluating {name} RelationNet...")
    suffix = name.lower().replace('+', '_')
    try:
        encoder.load_weights(f"best_relationnet_{suffix}_encoder.keras", by_name=True, skip_mismatch=True)
        rel_module.load_weights(f"best_relationnet_{suffix}_module.keras", by_name=True, skip_mismatch=True)
    except Exception as e:
        print(f"No saved weights found for {name}, using current weights.")
        
    relation_net = RelationNetwork(encoder, rel_module)
    trial_accs = []
    for i in range(100):
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        logits = relation_net(sx, sy, qx, 4, 5, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)
        
    mean_acc = np.mean(trial_accs)
    max_acc = np.max(trial_accs)
    min_acc = np.min(trial_accs)
    eval_results[name] = trial_accs
    
    print(f"5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")"""),
    ("code", """# --- 6. Plotting Cell ---
# Plotting training curves and evaluation results
histories = {
    'CNN': (loss_cnn, acc_cnn) if 'loss_cnn' in locals() else None,
    'CNN+Attention': (loss_attention, acc_attention) if 'loss_attention' in locals() else None,
    'CNN+Mamba': (loss_mamba, acc_mamba) if 'loss_mamba' in locals() else None,
    'CNN+Attention+Mamba': (loss_attn_mamba, acc_attn_mamba) if 'loss_attn_mamba' in locals() else None
}

# 1. Loss vs Episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[0], label=f'{name}')
plt.title('Loss vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy vs Episode
plt.subplot(1, 2, 2)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[1], label=f'{name}')
plt.title('Accuracy vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Holdout Evaluation Distributions
if eval_results:
    plt.figure(figsize=(10, 6))
    data_to_plot = [eval_results[name] for name in eval_results]
    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Test Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()""")
]

# ==================== MAML NOTEBOOK CELLS ====================
maml_cells = [
    ("markdown", "# Few-Shot Learning using Model-Agnostic Meta-Learning (MAML)\nThis notebook trains and evaluates four architectures (CNN, CNN+Attention, CNN+Mamba, CNN+Attention+Mamba) on the HyperLeaf spectral dataset using Model-Agnostic Meta-Learning (specifically First-Order MAML, FOMAML)."),
    ("code", """import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU configuration error:", e)

# Parameters
NUM_BANDS = 204
NUM_STATS = 5

# Load dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

# Stratified train/val/test split
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

# Function to sample episodic few-shot batches
def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=5):
    unique_classes = np.unique(y)
    selected_classes = unique_classes[:n_way]
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        sampled_indices = np.random.choice(class_indices, k_shot + q_query, replace=False)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.concatenate(query_y, axis=0)
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))

class MAMLClassifier(tf.keras.Model):
    def __init__(self, encoder, n_way=4):
        super(MAMLClassifier, self).__init__()
        self.encoder = encoder
        self.classifier_head = layers.Dense(n_way, activation=None)
    def call(self, x, training=True):
        embeddings = self.encoder(x, training=training)
        logits = self.classifier_head(embeddings)
        return logits

def train_step_fomaml(model, optimizer, support_x, support_y, query_x, query_y, alpha=0.01):
    original_weights = [tf.identity(v) for v in model.trainable_variables]
    with tf.GradientTape() as inner_tape:
        support_preds = model(support_x, training=True)
        loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            support_y, support_preds, from_logits=True
        ))
    grads_support = inner_tape.gradient(loss_support, model.trainable_variables)
    for var, grad in zip(model.trainable_variables, grads_support):
        if grad is not None:
            var.assign(var - alpha * grad)
            
    with tf.GradientTape() as outer_tape:
        query_preds = model(query_x, training=True)
        loss_query = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, query_preds, from_logits=True
        ))
    grads_outer = outer_tape.gradient(loss_query, model.trainable_variables)
    
    for var, orig_val in zip(model.trainable_variables, original_weights):
        var.assign(orig_val)
        
    trainable_vars, applied_grads = [], []
    for var, grad in zip(model.trainable_variables, grads_outer):
        if grad is not None:
            trainable_vars.append(var)
            applied_grads.append(grad)
    optimizer.apply_gradients(zip(applied_grads, trainable_vars))
    preds = tf.argmax(query_preds, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss_query, accuracy

def train_maml_model(maml_net, episodes=1000, lr=2e-4, alpha=0.01):
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_history, acc_history = [], []
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_fomaml(maml_net, optimizer, sx, sy, qx, qy, alpha=alpha)
        loss_history.append(loss.numpy())
        acc_history.append(acc.numpy())
        if ep % 200 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    return loss_history, acc_history"""),
    ("code", """# --- 1. CNN Architecture ---
def build_cnn_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x) # LN for MAML stability
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

cnn_encoder = build_cnn_encoder()
maml_cnn = MAMLClassifier(cnn_encoder, n_way=4)
print("Training CNN MAML...")
loss_cnn, acc_cnn = train_maml_model(maml_cnn, episodes=1000)
cnn_encoder.save("best_maml_cnn_encoder.keras")
maml_cnn.save_weights("best_maml_cnn_weights.h5")"""),
    ("code", """# --- 2. CNN + Attention Architecture ---
def build_cnn_attention_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    # Attention block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Encoder")

cnn_attention_encoder = build_cnn_attention_encoder()
maml_attention = MAMLClassifier(cnn_attention_encoder, n_way=4)
print("Training CNN+Attention MAML...")
loss_attention, acc_attention = train_maml_model(maml_attention, episodes=1000)
cnn_attention_encoder.save("best_maml_cnn_attention_encoder.keras")
maml_attention.save_weights("best_maml_cnn_attention_weights.h5")"""),
    ("code", """# --- 3. CNN + Mamba Architecture ---
class StableMambaBlock(layers.Layer):
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
            filters=self.d_inner, kernel_size=self.d_conv, padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        A_init = tf.keras.initializers.Constant(A_init_val)
        self.A = self.add_weight(
            name="A",
            shape=(self.d_inner, self.d_state),
            initializer=A_init,
            trainable=True
        )
        
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)
        
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(self.A, axis=0)
        A_expanded = tf.expand_dims(A_expanded, axis=0)
        
        A_bar = tf.exp(delta_expanded * A_expanded)
        
        B_expanded = tf.expand_dims(B, axis=2)
        B_bar = delta_expanded * B_expanded
        
        x_branch_expanded = tf.expand_dims(x_branch, axis=-1)
        inputs_ssm = B_bar * x_branch_expanded
        
        inputs_ssm = tf.transpose(inputs_ssm, perm=[1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        
        elems = (A_bar, inputs_ssm)
        batch_size = tf.shape(x)[0]
        h_0 = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=tf.float32)
        
        def scan_fn(h_prev, elem):
            a_bar_t, ssm_in_t = elem
            return a_bar_t * h_prev + ssm_in_t
        
        h_seq = tf.scan(scan_fn, elems, initializer=h_0)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])
        
        C_expanded = tf.expand_dims(C, axis=2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        
        D_expanded = tf.reshape(self.D, (1, 1, self.d_inner))
        y = y + D_expanded * x_branch
        
        gated_output = y * tf.keras.activations.swish(res_branch)
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

def build_cnn_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Mamba_Encoder")

cnn_mamba_encoder = build_cnn_mamba_encoder()
maml_mamba = MAMLClassifier(cnn_mamba_encoder, n_way=4)
print("Training CNN+Mamba MAML...")
loss_mamba, acc_mamba = train_maml_model(maml_mamba, episodes=1000)
cnn_mamba_encoder.save("best_maml_cnn_mamba_encoder.keras")
maml_mamba.save_weights("best_maml_cnn_mamba_weights.h5")"""),
    ("code", """# --- 4. CNN + Attention + Mamba Architecture ---
def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    
    mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Attention_Mamba_Encoder")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
maml_attn_mamba = MAMLClassifier(cnn_attention_mamba_encoder, n_way=4)
print("Training CNN+Attention+Mamba MAML...")
loss_attn_mamba, acc_attn_mamba = train_maml_model(maml_attn_mamba, episodes=1000)
cnn_attention_mamba_encoder.save("best_maml_cnn_attention_mamba_encoder.keras")
maml_attn_mamba.save_weights("best_maml_cnn_attention_mamba_weights.h5")"""),
    ("code", """# --- 5. Testing Cell (Holdout Evaluation) ---
# 5-Shot Holdout Evaluation over 100 trials for each model with adaptation
maml_models = {
    'CNN': maml_cnn,
    'CNN+Attention': maml_attention,
    'CNN+Mamba': maml_mamba,
    'CNN+Attention+Mamba': maml_attn_mamba
}

eval_results = {}

for name, maml_net in maml_models.items():
    print(f"\\nEvaluating {name} MAML Model (Test-Time Adaptation)...")
    suffix = name.lower().replace('+', '_')
    try:
        maml_net.load_weights(f"best_maml_{suffix}_weights.h5")
    except Exception as e:
        print(f"No saved weights found for {name}, using current weights.")
        
    trial_accs = []
    for i in range(100):
        # Sample test episode (5 support, 5 query per class)
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        
        # Save pre-adaptation weights
        original_weights = [tf.identity(v) for v in maml_net.trainable_variables]
        
        # Adaptation steps (5 steps fine-tuning on support set)
        for _ in range(5):
            with tf.GradientTape() as tape:
                preds_support = maml_net(sx, training=False)
                loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    sy, preds_support, from_logits=True
                ))
            grads = tape.gradient(loss_support, maml_net.trainable_variables)
            for var, grad in zip(maml_net.trainable_variables, grads):
                if grad is not None:
                    var.assign(var - 0.01 * grad)
                    
        logits = maml_net(qx, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)
        
        # Restore pre-adaptation weights
        for var, orig in zip(maml_net.trainable_variables, original_weights):
            var.assign(orig)
            
    mean_acc = np.mean(trial_accs)
    max_acc = np.max(trial_accs)
    min_acc = np.min(trial_accs)
    eval_results[name] = trial_accs
    
    print(f"5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")"""),
    ("code", """# --- 6. Plotting Cell ---
# Plotting training curves and evaluation results
histories = {
    'CNN': (loss_cnn, acc_cnn) if 'loss_cnn' in locals() else None,
    'CNN+Attention': (loss_attention, acc_attention) if 'loss_attention' in locals() else None,
    'CNN+Mamba': (loss_mamba, acc_mamba) if 'loss_mamba' in locals() else None,
    'CNN+Attention+Mamba': (loss_attn_mamba, acc_attn_mamba) if 'loss_attn_mamba' in locals() else None
}

# 1. Loss vs Episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[0], label=f'{name}')
plt.title('Loss vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy vs Episode
plt.subplot(1, 2, 2)
for name, hist in histories.items():
    if hist:
        plt.plot(hist[1], label=f'{name}')
plt.title('Accuracy vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Holdout Evaluation Distributions
if eval_results:
    plt.figure(figsize=(10, 6))
    data_to_plot = [eval_results[name] for name in eval_results]
    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Test Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()""")
]

# Generate notebooks
create_notebook("fsl_siemese.ipynb", siamese_cells)
create_notebook("fsl_protonet.ipynb", protonet_cells)
create_notebook("fsl_relationnet.ipynb", relationnet_cells)
create_notebook("fsl_maml.ipynb", maml_cells)
print("Notebook generation complete!")
