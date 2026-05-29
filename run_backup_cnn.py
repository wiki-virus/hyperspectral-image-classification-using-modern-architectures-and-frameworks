import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure GPU acceleration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

NUM_BANDS = 204
NUM_STATS = 5
CLASS_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']

# Load dataset
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
            idx1 = np.random.choice(class1_idx := class_indices[class1])
            idx2 = np.random.choice(class2_idx := class_indices[class2])
            pair_left.append(X[idx1])
            pair_right.append(X[idx2])
            labels.append(0.0)
    return [np.array(pair_left), np.array(pair_right)], np.array(labels)

X_train_pairs, y_train_pairs = generate_siamese_pairs(X_train, y_train, pairs_to_generate=10000)
X_val_pairs, y_val_pairs = generate_siamese_pairs(X_val, y_val, pairs_to_generate=2000)

BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_train_pairs[0], X_train_pairs[1]), y_train_pairs)
).shuffle(buffer_size=len(y_train_pairs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((X_val_pairs[0], X_val_pairs[1]), y_val_pairs)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_pure_cnn_encoder():
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    normalized_embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=normalized_embeddings, name="CNN_Encoder")

cnn_encoder = build_pure_cnn_encoder()
input_left = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="left_leaf")
input_right = layers.Input(shape=(NUM_BANDS, NUM_STATS), name="right_leaf")
encoded_left = cnn_encoder(input_left)
encoded_right = cnn_encoder(input_right)
l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_left, encoded_right])
prediction = layers.Dense(1, activation='sigmoid')(l1_distance)
siamese_model = models.Model(inputs=[input_left, input_right], outputs=prediction)
siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_siamese_fewshot_backup.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
)

print("Starting Siamese training...")
with tf.device("/GPU:0"):
    history = siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[checkpoint, early_stop]
    )

def run_few_shot_inference(encoder, support_X, support_y, test_X, test_y, n_shots=3):
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

print("Evaluating...")
predicted_labels = run_few_shot_inference(
    encoder=cnn_encoder,
    support_X=X_train,
    support_y=y_train,
    test_X=X_test,
    test_y=y_test,
    n_shots=3
)
acc = np.mean(predicted_labels == y_test) * 100
print(f"Few-shot accuracy on test set: {acc:.2f}%")
