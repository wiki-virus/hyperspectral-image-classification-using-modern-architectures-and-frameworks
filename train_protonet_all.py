import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configure GPU
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
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

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

def build_encoder(arch_type, embedding_dim=32, l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    if arch_type == 'cnn':
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
    elif arch_type == 'cnn_attention':
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
    elif arch_type == 'cnn_mamba':
        mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
        x = layers.Add()([x, mamba_out])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
    elif arch_type == 'cnn_attention_mamba':
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
    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=embeddings, name=f"Encoder_{arch_type}")

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

architectures = ['cnn', 'cnn_attention', 'cnn_mamba', 'cnn_attention_mamba']
results = {}

for arch in architectures:
    print(f"\n==================== TRAINING {arch.upper()} PROTONET ====================")
    encoder = build_encoder(arch)
    proto_net = PrototypicalNetwork(encoder)
    optimizer = tf.keras.optimizers.Adam(2e-4) # Tuning optimizer to 2e-4
    
    loss_history = []
    acc_history = []
    
    # Train for 3000 episodes
    for ep in range(1, 3001):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_protonet(proto_net, optimizer, sx, sy, qx, qy, 4, 5)
        loss_history.append(loss.numpy())
        acc_history.append(acc.numpy())
        if ep % 500 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
            
    # Save encoder
    encoder.save(f"best_protonet_{arch}.keras")
    
    # Evaluate over 100 trials
    trial_accs = []
    for i in range(100):
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        # For evaluation, run without dropout/noise
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
    
    print(f"==> {arch.upper()} ProtoNet 5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")
    
    results[arch] = {
        'loss': loss_history,
        'accuracy': acc_history,
        'trial_accs': trial_accs,
        'mean_acc': mean_acc,
        'max_acc': max_acc,
        'min_acc': min_acc
    }

# Save results dictionary as npz file
np.savez("protonet_results.npz", **results)
print("\nAll ProtoNet architectures trained and saved!")
