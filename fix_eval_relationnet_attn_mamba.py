"""Quick fix: Re-evaluate RelationNet CNN+Attn+Mamba with the correct saved weights."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3800)]
        )
    except Exception as e:
        print("GPU config error:", e)

NUM_BANDS = 204
NUM_STATS = 5
EMBED_DIM = 64
N_WAY = 4
K_SHOT = 5

loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=15):
    unique_classes = np.unique(y)
    selected_classes = np.random.choice(unique_classes, size=n_way, replace=False)
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        needed = k_shot + q_query
        replace = len(class_indices) < needed
        sampled_indices = np.random.choice(class_indices, needed, replace=replace)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x)
    support_y = np.concatenate(support_y)
    query_x   = np.concatenate(query_x)
    query_y   = np.concatenate(query_y)
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x,   dtype=tf.float32),
            tf.convert_to_tensor(query_y,   dtype=tf.int32))

class StableMambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.d_state = d_state
        self.d_conv = d_conv; self.expand = expand
        self.d_inner = int(expand * d_model)
    def build(self, input_shape):
        self.in_proj  = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d   = layers.Conv1D(filters=self.d_inner, kernel_size=self.d_conv,
                                      padding='same', groups=self.d_inner, activation='swish')
        self.x_proj   = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj  = layers.Dense(self.d_inner, activation='softplus')
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        self.A = self.add_weight(name="A", shape=(self.d_inner, self.d_state),
                                 initializer=tf.keras.initializers.Constant(A_init_val), trainable=True)
        self.D = self.add_weight(name="D", shape=(self.d_inner,),
                                 initializer=tf.keras.initializers.Ones(), trainable=True)
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)
    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, 2, axis=-1)
        x_branch = self.conv1d(x_branch)
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(tf.expand_dims(self.A, 0), 0)
        A_bar = tf.exp(delta_expanded * A_expanded)
        B_bar = delta_expanded * tf.expand_dims(B, 2)
        inputs_ssm = B_bar * tf.expand_dims(x_branch, -1)
        inputs_ssm = tf.transpose(inputs_ssm, [1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, [1, 0, 2, 3])
        h_0 = tf.zeros((tf.shape(x)[0], self.d_inner, self.d_state), dtype=tf.float32)
        def scan_fn(h, elem):
            a, s = elem
            return a * h + s
        h_seq = tf.scan(scan_fn, (A_bar, inputs_ssm), initializer=h_0)
        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])
        y = tf.reduce_sum(h_seq * tf.expand_dims(C, 2), axis=-1)
        y = y + tf.reshape(self.D, (1, 1, self.d_inner)) * x_branch
        return self.out_proj(y * tf.keras.activations.swish(res_branch))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "d_state": self.d_state,
                    "d_conv": self.d_conv, "expand": self.expand})
        return cfg

def build_cnn_attention_mamba_encoder(activation='leaky_relu', l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(128, kernel_size=7, activation=activation, padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=5, activation=activation, padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    mamba = StableMambaBlock(d_model=256, d_state=32, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation=activation, padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=activation, kernel_regularizer=reg)(x)
    out = layers.Dense(EMBED_DIM, activation=None)(x)
    return models.Model(inputs=inputs, outputs=out, name="CNN_AttnMamba_Encoder_REL_OPT")

def build_relation_module(embed_dim=64, l2_reg=1e-4):
    inputs = layers.Input(shape=(embed_dim * 2,))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=reg)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=out, name="Relation_Module_OPT")

# Build models
enc = build_cnn_attention_mamba_encoder()
rel = build_relation_module(embed_dim=EMBED_DIM)

enc_file = "best_opt_relationnet_cnn_attention_mamba_encoder.keras"
rel_file = "best_opt_relationnet_cnn_attention_mamba_module.keras"

enc.load_weights(enc_file)
rel.load_weights(rel_file)
print(f"Loaded weights from {enc_file} and {rel_file}")

# Evaluate 100 trials
trial_accs = []
for i in range(100):
    sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, N_WAY, K_SHOT, 15)
    sup_emb = enc(sx, training=False)
    qry_emb = enc(qx, training=False)
    
    # Compute prototypes
    d = tf.shape(sup_emb)[-1]
    protos = tf.reduce_mean(tf.reshape(sup_emb, (N_WAY, K_SHOT, d)), axis=1)
    
    # Evaluate relation
    q_len = tf.shape(qry_emb)[0]
    p_len = tf.shape(protos)[0]
    
    qry_rep = tf.repeat(qry_emb, p_len, axis=0)
    proto_rep = tf.tile(protos, [q_len, 1])
    
    concat_pairs = tf.concat([proto_rep, qry_rep], axis=1)
    rel_scores = rel(concat_pairs, training=False)
    rel_scores = tf.reshape(rel_scores, (q_len, p_len))
    
    preds = tf.argmax(rel_scores, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
    trial_accs.append(acc * 100.0)

print(f"\\nRelationNet CNN+Attn+Mamba (CORRECTED holdout):")
print(f"  Mean: {np.mean(trial_accs):.2f}%")
print(f"  Max:  {np.max(trial_accs):.2f}%")
print(f"  Min:  {np.min(trial_accs):.2f}%")
print(f"  Std:  {np.std(trial_accs):.2f}%")
