# === CELL 1 ===
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
        print(f"[GPU] Detected and Configured: {gpus}\n")
    except RuntimeError as e:
        print(f"[WARN] GPU already initialized: {e}\n")
else:
    print(f"[WARN] No GPU found. Falling back to CPU.\n")

# Global configuration parameters
NUM_BANDS = 204
NUM_STATS = 5
CLASS_NAMES = ['Heerup', 'Kvium', 'Rembrandt', 'Sheriff']
NUM_CLASSES = 4


# === CELL 3 ===
# Load dataset
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']
print(f"Source tensor shape: {X_data.shape}")

# Stratified splitting (80% train, 10% val, 10% test)
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalization
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_val = (X_val_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

print(f"Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")


# === CELL 5 ===
def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=5):
    """
    Generates a single FSL episode containing support and query sets.
    """
    unique_classes = np.unique(y)
    # For this dataset, we perform N-way where N matches the number of unique classes
    selected_classes = unique_classes[:n_way]
    
    support_x, support_y = [], []
    query_x, query_y = [], []
    
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
    
    # Shuffle query set to prevent order bias
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))


# === CELL 7 ===
def build_backbone_encoder(embedding_dim=32, norm_type='batch', l2_normalize=True, activation='relu', name="CNN_Backbone_Encoder"): 
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    
    # Layer 1
    x = layers.Conv1D(64, kernel_size=5, activation=None, padding='same')(inputs)
    if activation == 'relu':
        x = layers.ReLU()(x)
    elif activation == 'leaky_relu':
        x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    if norm_type == 'batch':
        x = layers.BatchNormalization()(x)
    elif norm_type == 'layer':
        x = layers.LayerNormalization()(x)
    
    # Layer 2
    x = layers.Conv1D(128, kernel_size=5, activation=None, padding='same')(x)
    if activation == 'relu':
        x = layers.ReLU()(x)
    elif activation == 'leaky_relu':
        x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    if norm_type == 'batch':
        x = layers.BatchNormalization()(x)
    elif norm_type == 'layer':
        x = layers.LayerNormalization()(x)
    
    # Layer 3
    x = layers.Conv1D(256, kernel_size=5, activation=None, padding='same')(x)
    if activation == 'relu':
        x = layers.ReLU()(x)
    elif activation == 'leaky_relu':
        x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    if norm_type == 'batch':
        x = layers.BatchNormalization()(x)
    elif norm_type == 'layer':
        x = layers.LayerNormalization()(x)
    
    # Dense projection & L2 normalization
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=None)(x)
    if activation == 'relu':
        x = layers.ReLU()(x)
    elif activation == 'leaky_relu':
        x = layers.LeakyReLU(alpha=0.1)(x)
    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    if l2_normalize:
        embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    else:
        embeddings = raw_embeddings
    
    return models.Model(inputs=inputs, outputs=embeddings, name=name)

backbone_test = build_backbone_encoder()
backbone_test.summary()


# === CELL 9 ===
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder

    def call(self, support_x, support_y, query_x, n_way, k_shot):
        # Encode support & query sets
        support_embeddings = self.encoder(support_x, training=True)
        query_embeddings = self.encoder(query_x, training=True)
        
        # Calculate class prototypes
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1) # (n_way, D)
        
        # Compute distance to prototypes
        query_expanded = tf.expand_dims(query_embeddings, axis=1)  # (N*Q, 1, D)
        proto_expanded = tf.expand_dims(prototypes, axis=0)        # (1, N, D)
        distances = tf.reduce_sum(tf.square(query_expanded - proto_expanded), axis=-1)  # (N*Q, N)
        
        return -distances # Return negative distances as logits

# Helper training function compiled with XLA
@tf.function(jit_compile=True)
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


# === CELL 11 ===
def build_relation_module(embedding_dim=32):
    inputs = layers.Input(shape=(4 * embedding_dim,))
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(64, activation=None)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(32, activation=None)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    outputs = layers.Dense(1, activation=None)(x) # Output logits, no activation
    return models.Model(inputs=inputs, outputs=outputs, name="Relation_Module")

class RelationNetwork(tf.keras.Model):
    def __init__(self, encoder, relation_module):
        super(RelationNetwork, self).__init__()
        self.encoder = encoder
        self.relation_module = relation_module

    def call(self, support_x, support_y, query_x, n_way, k_shot):
        support_embeddings = self.encoder(support_x, training=True)
        query_embeddings = self.encoder(query_x, training=True)
        
        # Calculate prototypes
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1) # (n_way, D)
        
        num_queries = tf.shape(query_embeddings)[0]
        
        # Pairwise concatenation
        query_expanded = tf.expand_dims(query_embeddings, axis=1)  # (N*Q, 1, D)
        query_tiled = tf.tile(query_expanded, [1, n_way, 1])        # (N*Q, N, D)
        
        proto_expanded = tf.expand_dims(prototypes, axis=0)        # (1, N, D)
        proto_tiled = tf.tile(proto_expanded, [num_queries, 1, 1])  # (N*Q, N, D)
        
        # Rich relation input: [q, p, |q-p|, q*p]
        diff = tf.abs(query_tiled - proto_tiled)
        prod = query_tiled * proto_tiled
        relation_input = tf.concat([query_tiled, proto_tiled, diff, prod], axis=-1)  # (N*Q, N, 4*D)
        
        # Flatten for relation module input
        flat_shape = tf.shape(relation_input)
        flat_input = tf.reshape(relation_input, (flat_shape[0] * flat_shape[1], flat_shape[2]))
        
        relation_logits = self.relation_module(flat_input, training=True)  # (N*Q * N, 1)
        relation_logits = tf.reshape(relation_logits, (flat_shape[0], flat_shape[1])) # (N*Q, N)
        return relation_logits

# Helper training function compiled with XLA
@tf.function(jit_compile=True)
def train_step_relationnet(model, optimizer, support_x, support_y, query_x, query_y, n_way, k_shot):
    with tf.GradientTape() as tape:
        relation_logits = model(support_x, support_y, query_x, n_way, k_shot)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, relation_logits, from_logits=True
        )) # SCCE Loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    preds = tf.argmax(relation_logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, accuracy


# === CELL 13 ===
class MAMLClassifier(tf.keras.Model):
    def __init__(self, encoder, n_way):
        super(MAMLClassifier, self).__init__()
        self.encoder = encoder
        self.classifier_head = layers.Dense(n_way, activation=None)

    def call(self, x, training=True):
        embeddings = self.encoder(x, training=training)
        logits = self.classifier_head(embeddings)
        return logits

# Helper training function using @tf.function (non-JIT due to variable assignments)
@tf.function
def train_step_fomaml(model, optimizer, support_x, support_y, query_x, query_y, alpha=0.01):
    # Save current weights as tensors to restore after inner loop update
    original_weights = [tf.identity(v) for v in model.trainable_variables]
    
    # 1. Inner Loop (Task adaptation step)
    with tf.GradientTape() as inner_tape:
        support_preds = model(support_x, training=True)
        loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            support_y, support_preds, from_logits=True
        ))
    grads_support = inner_tape.gradient(loss_support, model.trainable_variables)
    
    # Apply gradient descent update directly to variables
    for var, grad in zip(model.trainable_variables, grads_support):
        if grad is not None:
            var.assign(var - alpha * grad)
            
    # 2. Outer Loop (Meta-gradient computation on query set)
    with tf.GradientTape() as outer_tape:
        query_preds = model(query_x, training=True)
        loss_query = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, query_preds, from_logits=True
        ))
    grads_outer = outer_tape.gradient(loss_query, model.trainable_variables)
    
    # 3. Restore original weights
    for var, orig_val in zip(model.trainable_variables, original_weights):
        var.assign(orig_val)
        
    # 4. Meta-update
    trainable_vars = []
    applied_grads = []
    for var, grad in zip(model.trainable_variables, grads_outer):
        if grad is not None:
            trainable_vars.append(var)
            applied_grads.append(grad)
            
    optimizer.apply_gradients(zip(applied_grads, trainable_vars))
    
    # Calculate accuracy
    preds = tf.argmax(query_preds, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss_query, accuracy


# === CELL 15 ===
import gc

# Training configuration
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
NUM_EPISODES = 2000
LEARNING_RATE = 1e-3

# Clear Keras VRAM
tf.keras.backend.clear_session()
gc.collect()

# 1. Initialize Prototypical Network
print("Training Prototypical Network...")
proto_encoder = build_backbone_encoder(norm_type='batch', l2_normalize=True, activation='relu', name="ProtoNet_Encoder")
proto_net = PrototypicalNetwork(proto_encoder)
proto_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

for episode in range(1, NUM_EPISODES + 1):
    support_x, support_y, query_x, query_y = get_few_shot_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
    loss, acc = train_step_protonet(
        proto_net, proto_optimizer, support_x, support_y, query_x, query_y, N_WAY, K_SHOT
    )
    if episode % 200 == 0:
        print(f"  Episode {episode:04d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# 2. Initialize Relation Network
print("\nTraining Relation Network...")
relation_encoder = build_backbone_encoder(norm_type='batch', l2_normalize=True, activation='leaky_relu', name="RelationNet_Encoder")
relation_module = build_relation_module()
relation_net = RelationNetwork(relation_encoder, relation_module)
relation_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

for episode in range(1, NUM_EPISODES + 1):
    support_x, support_y, query_x, query_y = get_few_shot_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
    loss, acc = train_step_relationnet(
        relation_net, relation_optimizer, support_x, support_y, query_x, query_y, N_WAY, K_SHOT
    )
    if episode % 200 == 0:
        print(f"  Episode {episode:04d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# 3. Initialize FOMAML Network
print("\nTraining FOMAML Network...")
maml_encoder = build_backbone_encoder(norm_type='layer', l2_normalize=True, activation='leaky_relu', name="MAML_Encoder")
maml_net = MAMLClassifier(maml_encoder, N_WAY)
maml_net_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

for episode in range(1, NUM_EPISODES + 1):
    support_x, support_y, query_x, query_y = get_few_shot_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
    loss, acc = train_step_fomaml(
        maml_net, maml_net_optimizer, support_x, support_y, query_x, query_y, alpha=0.01
    )
    if episode % 200 == 0:
        print(f"  Episode {episode:04d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")


# === CELL 17 ===
print("Evaluating and comparing all three FSL frameworks on 100 test episodes...")
EVAL_EPISODES = 100

proto_accs = []
relation_accs = []
maml_accs = []

for i in range(EVAL_EPISODES):
    # Get evaluation episode from testing set
    support_x, support_y, query_x, query_y = get_few_shot_episode(X_test, y_test, N_WAY, K_SHOT, Q_QUERY)
    
    # 1. ProtoNet Eval
    proto_logits = proto_net(support_x, support_y, query_x, N_WAY, K_SHOT)
    proto_preds = tf.argmax(proto_logits, axis=-1, output_type=tf.int32)
    proto_acc = tf.reduce_mean(tf.cast(tf.equal(proto_preds, query_y), tf.float32)).numpy()
    proto_accs.append(proto_acc)
    
    # 2. RelationNet Eval
    relation_scores = relation_net(support_x, support_y, query_x, N_WAY, K_SHOT)
    relation_preds = tf.argmax(relation_scores, axis=-1, output_type=tf.int32)
    relation_acc = tf.reduce_mean(tf.cast(tf.equal(relation_preds, query_y), tf.float32)).numpy()
    relation_accs.append(relation_acc)
    
    # 3. MAML Eval (Requires adaptation step for each test episode)
    # Save current weights
    original_maml_weights = [tf.identity(v) for v in maml_net.trainable_variables]
    
    # Inner adaptation loop (we use a few adaptation steps at test-time, say 5 steps)
    for _ in range(5):
        with tf.GradientTape() as inner_tape:
            support_preds = maml_net(support_x, training=False)
            loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                support_y, support_preds, from_logits=True
            ))
        grads = inner_tape.gradient(loss_support, maml_net.trainable_variables)
        for var, grad in zip(maml_net.trainable_variables, grads):
            if grad is not None:
                var.assign(var - 0.01 * grad)
                
    # Evaluate on query set after adaptation
    maml_logits = maml_net(query_x, training=False)
    maml_preds = tf.argmax(maml_logits, axis=-1, output_type=tf.int32)
    maml_acc = tf.reduce_mean(tf.cast(tf.equal(maml_preds, query_y), tf.float32)).numpy()
    maml_accs.append(maml_acc)
    
    # Restore original weights so model is ready for the next test episode
    for var, orig_val in zip(maml_net.trainable_variables, original_maml_weights):
        var.assign(orig_val)

print("\n" + "=" * 60)
print("FINAL ACCURACY RESULTS (Average over 100 Episodes):")
print("=" * 60)
print(f"Prototypical Network (ProtoNet):  {np.mean(proto_accs)*100:6.2f}%")
print(f"Relation Network (RelationNet):  {np.mean(relation_accs)*100:6.2f}%")
print(f"First-Order MAML (FOMAML):      {np.mean(maml_accs)*100:6.2f}%")
print("=" * 60)

