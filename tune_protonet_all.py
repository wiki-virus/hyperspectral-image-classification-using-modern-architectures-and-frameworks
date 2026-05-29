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

class MambaBlock(layers.Layer):
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

def build_encoder(arch_type, embedding_dim=32):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    
    # Base Conv block 1
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Base Conv block 2
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Architecture branches
    if arch_type == 'cnn':
        # Simple third Conv1D
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
    elif arch_type == 'cnn_attention':
        # Attention
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)
        
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
    elif arch_type == 'cnn_mamba':
        # Mamba
        mamba_out = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
        x = layers.Add()([x, mamba_out])
        x = layers.LayerNormalization()(x)
        
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
    elif arch_type == 'cnn_attention_mamba':
        # Attention + Mamba
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)
        
        mamba_out = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
        x = layers.Add()([x, mamba_out])
        x = layers.LayerNormalization()(x)
        
        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
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
for arch in architectures:
    print(f"\n--- Training ProtoNet on {arch} ---")
    encoder = build_encoder(arch)
    proto_net = PrototypicalNetwork(encoder)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    # Train for 3000 episodes
    for ep in range(1, 3001):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_protonet(proto_net, optimizer, sx, sy, qx, qy, 4, 5)
        if ep % 1000 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
            
    # Evaluate
    accs = []
    for i in range(100):
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        logits = proto_net(sx, sy, qx, 4, 5)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        accs.append(acc)
    print(f"==> {arch} test accuracy: {np.mean(accs)*100:.2f}%")
